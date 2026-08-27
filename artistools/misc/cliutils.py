"""Shared helpers for command-line argument parsing and list/path argument normalisation."""

import argparse
import itertools
import sys
import typing as t
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from pathlib import Path

from artistools.commands import CustomArgHelpFormatter
from artistools.commands import SuggestingArgumentParser

if t.TYPE_CHECKING:
    from collections.abc import Collection

    import numpy as np
    import numpy.typing as npt

# a path argument arrives as a scalar, as a sequence, or as a nested sequence from repeated -modelpath
type PathArg = Path | str | Sequence[PathArg] | None


def arggroup(parser: argparse.ArgumentParser, title: str) -> "argparse._ArgumentGroup":  # pyright: ignore[reportPrivateUsage]
    """Return the argument group of the parser with this title, and make it if the parser has none.

    The flagship commands hold more than 70 options, thus a flat listing is hard to read. Each shared
    helper puts its arguments into a titled group, and the help of every command gains the same shape.
    """
    for group in parser._action_groups:  # ruff:ignore[private-member-access]  # pyright: ignore[reportPrivateUsage]
        if group.title == title:
            return group

    return parser.add_argument_group(title)


def addarg_viewingangle(parser: argparse.ArgumentParser, allow_select_all: bool = False) -> None:
    """Add the viewing direction selection and averaging arguments shared by the plotting commands."""
    parser.add_argument(
        "-plotvspecpol",
        type=int,
        metavar="n",
        nargs="+",
        help="Plot viewing angles from vspecpol virtual packets. Expects int for angle = spec number in vspecpol files",
    )

    parser.add_argument(
        "-plotviewingangle",
        "-dirbin",
        type=int,
        metavar="n",
        nargs="+",
        help=(
            "Plot viewing directions. Expects int for direction bin in specpol_res.out"
            + (". Use -2 to select all viewing angles" if allow_select_all else "")
        ),
    )

    parser.add_argument(
        "--usedegrees",
        action="store_true",
        help="Use degrees instead of radians for direction angles. Only works with -plotviewingangle",
    )

    # averaging over one angle leaves one bin per index of the other, so the two cannot be combined. argparse
    # enforces this once here for every command that takes these flags
    averagegroup = parser.add_mutually_exclusive_group()

    averagegroup.add_argument(
        "--average_over_phi_angle",
        action="store_true",
        help="Average over phi (azimuthal) viewing angles to make direction bins into polar angle bins",
    )

    # deprecated alias for --average_over_phi_angle kept for backwards compatibility
    averagegroup.add_argument("--average_every_tenth_viewing_angle", action="store_true", help=argparse.SUPPRESS)

    averagegroup.add_argument(
        "--average_over_theta_angle",
        action="store_true",
        help="Average over theta (polar) viewing angles to make direction bins into azimuthal angle bins",
    )


class KeepGivenPaths(argparse.Action):
    """Store the paths of a positional argument, but keep the paths that the option form already gave.

    argparse applies a positional after an option that shares its dest, thus a positional that the user
    left out would otherwise hide the value of that option. argparse gives the default of the positional
    as the value in that case, thus a value equal to that default counts as no value at all.
    """

    def __call__(
        self,
        parser: argparse.ArgumentParser,  # ruff:ignore[unused-method-argument]
        namespace: argparse.Namespace,
        values: "str | Sequence[t.Any] | None",
        option_string: str | None = None,  # ruff:ignore[unused-method-argument]
    ) -> None:
        """Set the paths of the positional argument, unless the option form already gave some."""
        userwrote = bool(values) and values != self.default
        if userwrote or getattr(namespace, self.dest, None) is None:
            setattr(namespace, self.dest, values)


def addarg_pathoption(parser: argparse.ArgumentParser, flag: str, dest: str, *, multiplepaths: bool) -> None:
    """Accept an option that names the same paths as a positional argument.

    Some commands take the paths as a positional argument and others take -modelpath. A user who learns
    one form must not meet "unrecognized arguments" with the other, thus the option stands beside the
    positional. It stays out of the help text, because the positional already gives the paths a name.
    """
    optionkwargs: dict[str, t.Any] = {
        "dest": dest,
        "type": Path,
        "default": argparse.SUPPRESS,
        "help": argparse.SUPPRESS,
    }
    if multiplepaths:
        optionkwargs["nargs"] = "*"

    parser.add_argument(flag, **optionkwargs)


def addarg_modelpath(
    parser: argparse.ArgumentParser,
    *,
    positional: bool = False,
    multiplepaths: bool = False,
    required: bool = False,
    default: t.Any = None,
    helptext: str = "Path to ARTIS folder",
) -> None:
    """Add the ARTIS model path argument (-modelpath option, or a positional path when positional=True)."""
    kwargs: dict[str, t.Any] = {"type": Path, "default": default, "help": helptext}
    if multiplepaths:
        kwargs["nargs"] = "*"
    if positional:
        parser.add_argument("modelpath", action=KeepGivenPaths, **kwargs)
        addarg_pathoption(parser, "-modelpath", "modelpath", multiplepaths=multiplepaths)
    else:
        if required:
            kwargs["required"] = True
        parser.add_argument("-modelpath", **kwargs)


def addarg_output(
    parser: argparse.ArgumentParser,
    *,
    kind: t.Literal["file", "folder"],
    defaultname: str | None = None,
    default: t.Any = None,
    astype: type[Path] | type[str] | None = Path,
    extraflags: "Sequence[str]" = (),
    helptext: str | None = None,
) -> None:
    """Add the -outputfile/-o argument, and record what the command writes.

    A command writes one file or a folder of files, and kind says which. resolve_output_argument reads
    that word after the parse: it gives a file the defaultname of the command when -o names a folder,
    and it makes the folder that -o names either way. Thus every command keeps the promise of the help
    text, and no command writes that rule again.

    A command that names its own frames takes no defaultname, because resolve_frameset_paths gives each
    frame a name of its own.
    """
    rule = (
        "A path with no file extension names a folder, which the command creates"
        if kind == "file"
        else "The command creates this folder"
    )
    kwargs: dict[str, t.Any] = {
        "dest": "outputfile",
        "default": default,
        "help": f"{helptext or ('Path/filename for the output file' if kind == 'file' else 'Path for the output files')}. {rule}",
    }
    if astype is not None:
        kwargs["type"] = astype

    arggroup(parser, "output").add_argument("-outputfile", *extraflags, "-outputpath", "-o", **kwargs)
    parser.set_defaults(outputkind=kind, outputdefaultname=defaultname)


def resolve_output_argument(args: argparse.Namespace) -> None:
    """Apply the rule of -o that addarg_output recorded on the parser of the command.

    A command that writes one file takes the name of that file when -o names a folder. A command that
    writes a folder of files gets that folder. The folder exists after this either way.
    """
    outputfile = getattr(args, "outputfile", None)
    kind = getattr(args, "outputkind", None)
    if kind is None or not outputfile:
        return

    if kind == "folder":
        Path(outputfile).mkdir(parents=True, exist_ok=True)
    elif (defaultname := getattr(args, "outputdefaultname", None)) is not None:
        args.outputfile = resolve_outputfile(outputfile, defaultname)


def addarg_modelgridindex(
    parser: argparse.ArgumentParser,
    *,
    kind: t.Literal["rangestr", "int", "append"] = "int",
    default: t.Any = None,
    helptext: str | None = None,
) -> None:
    """Add the -modelgridindex/-cell/-mgi argument that selects the model grid cell or cells.

    A command that plots one cell takes kind="int". A command that reads several cells takes a range
    string such as 3-7, which parse_range_list expands. Every command offers the same three flags.
    """
    group = arggroup(parser, "cell selection")
    flags = ("-modelgridindex", "-cell", "-mgi")
    helptext = helptext or "Model grid cell to plot"
    if kind == "int":
        group.add_argument(*flags, type=int, default=default, help=helptext)
    elif kind == "append":
        group.add_argument(*flags, action="append", default=default, help=helptext)
    else:
        group.add_argument(*flags, default=default, help=helptext)


class UnsupportedArgument(argparse.Action):
    """Stop the command, and name the argument to give in place of the one that the user gave."""

    def __init__(self, option_strings: "Sequence[str]", dest: str, instead: str = "", **kwargs: t.Any) -> None:
        """Take the name of the argument that this command does take."""
        super().__init__(option_strings, dest, nargs="?", help=argparse.SUPPRESS, **kwargs)
        self.instead = instead

    @t.override
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: "str | Sequence[t.Any] | None",
        option_string: str | None = None,
    ) -> None:
        """Report that this command does not take the argument."""
        helptext = f"Give {self.instead} instead" if self.instead else ""
        if not helptext and isinstance(parser, SuggestingArgumentParser):
            helptext = suggest_names(str(option_string), parser.get_visible_flags())

        if isinstance(parser, SuggestingArgumentParser):
            parser.exit_with_help(
                f"{option_string} is not an argument of this command",
                helptext or f"Run `{parser.prog} --help` to see every argument",
            )

        parser.error(f"{option_string} is not an argument of this command. {helptext}".rstrip())


def addarg_collidingflags(parser: argparse.ArgumentParser) -> None:
    """Declare the flag names of other commands that this command would read as a joined value.

    argparse joins a value to a flag of one letter, thus "-obsspec 100" on a command that takes -o but
    no -obsspec reads as "-o bsspec" and writes the plot to a file named bsspec. A declared name gives
    a message in place of that.

    An exact name comes before a prefix for argparse, thus a declared name keeps every flag of this
    command and every abbreviation of one. A measurement over the tree gives the same 2208 abbreviations
    with these names and without them.
    """
    from artistools.commands import SINGLEDASHLONGFLAGS

    declared = {flag for action in parser._actions for flag in action.option_strings}  # ruff:ignore[private-member-access]
    oneletter = {flag for flag in declared if len(flag) == 2 and not flag.startswith("--")}

    for name in sorted(SINGLEDASHLONGFLAGS):
        collides = any(name.startswith(letterflag) for letterflag in oneletter)
        if collides and name not in declared:
            parser.add_argument(name, action=UnsupportedArgument, default=argparse.SUPPRESS)


def addarg_unsupported(parser: argparse.ArgumentParser, *flags: str, instead: str) -> None:
    """Declare an argument that this command does not take, so that a user gets a clear message.

    argparse joins a value to a single-dash flag, thus "-timestep 30" on a parser that declares -t but
    no -timestep reads as "-t imestep". A declared name gives a message that names the right argument.
    """
    parser.add_argument(*flags, action=UnsupportedArgument, instead=instead, default=argparse.SUPPRESS)


def addarg_timestep(
    parser: argparse.ArgumentParser,
    *,
    kind: t.Literal["rangestr", "int", "strappend"] = "rangestr",
    default: t.Any = None,
    helptext: str | None = None,
) -> None:
    """Add the -timestep/-ts argument: a range string like 45-65, a single int, or an appendable list."""
    group = arggroup(parser, "time selection")
    flags = ("-timestep", "-ts")
    if kind == "rangestr":
        group.add_argument(
            *flags, dest="timestep", nargs="?", default=default, help=helptext or "First timestep or a range e.g. 45-65"
        )
    elif kind == "int":
        group.add_argument(*flags, type=int, default=default, help=helptext or "Timestep number to plot")
    else:
        group.add_argument(*flags, action="append", default=default, help=helptext or "Timestep number to plot")


def addarg_timedays(
    parser: argparse.ArgumentParser,
    *,
    kind: t.Literal["rangestr", "str", "float"] = "rangestr",
    helptext: str | None = None,
) -> None:
    """Add the -timedays/-time/-t argument, either as a range string like 50-100 or a single value.

    -t means -timedays on every command, thus a user needs no knowledge of which other arguments a
    command takes. A command that takes no -timestep calls addarg_unsupported for that name, because
    argparse joins a value to a single-dash flag and would read "-timestep 30" as "-t imestep".
    """
    group = arggroup(parser, "time selection")
    flags = ("-timedays", "-time", "-t")
    if kind == "rangestr":
        group.add_argument(
            *flags, dest="timedays", nargs="?", help=helptext or "Range of times in days to plot (e.g. 50-100)"
        )
    elif kind == "float":
        group.add_argument(*flags, type=float, help=helptext or "Time in days to plot")
    else:
        group.add_argument(*flags, help=helptext or "Time in days to plot")


def addarg_timeminmax(
    parser: argparse.ArgumentParser,
    *,
    helptext_min: str = "Lower time in days",
    helptext_max: str = "Upper time in days",
) -> None:
    """Add the -timemin and -timemax arguments bounding a time range in days."""
    group = arggroup(parser, "time selection")
    group.add_argument("-timemin", type=float, help=helptext_min)
    group.add_argument("-timemax", type=float, help=helptext_max)


def addarg_axislimits(
    parser: argparse.ArgumentParser,
    *,
    xlimtype: type[int] | type[float] = float,
    xmindefault: float | None = None,
    xmaxdefault: float | None = None,
    xminhelp: str = "Plot range: minimum x value",
    xmaxhelp: str = "Plot range: maximum x value",
    include_x: bool = True,
    include_y: bool = True,
    wavelength_aliases: bool = False,
) -> None:
    """Add the -xmin/-xmax and -ymin/-ymax plot range arguments.

    A command whose x axis is a wavelength in Angstroms takes wavelength_aliases, which adds the
    -lambdamin and -lambdamax spellings of the same arguments.
    """
    group = arggroup(parser, "appearance")
    if include_x:
        xminflags = ("-xmin", "-lambdamin") if wavelength_aliases else ("-xmin",)
        xmaxflags = ("-xmax", "-lambdamax") if wavelength_aliases else ("-xmax",)
        group.add_argument(*xminflags, dest="xmin", type=xlimtype, default=xmindefault, help=xminhelp)
        group.add_argument(*xmaxflags, dest="xmax", type=xlimtype, default=xmaxdefault, help=xmaxhelp)
    if include_y:
        group.add_argument("-ymin", type=float, default=None, help="Plot range: y-axis minimum")
        group.add_argument("-ymax", type=float, default=None, help="Plot range: y-axis maximum")


def color_arg(value: str) -> str:
    """Return a colour the user asked for, rejecting one matplotlib cannot parse.

    The colours are compared and resolved long before anything is drawn, so a typo caught here names the
    argument that caused it instead of surfacing from inside a plotting helper.
    """
    import matplotlib.colors as mplcolors

    if not mplcolors.is_color_like(value):
        msg = f"not a matplotlib color: {value}"
        raise argparse.ArgumentTypeError(msg)

    return value


def addarg_seriesstyle(
    parser: argparse.ArgumentParser,
    *,
    colordefault: Sequence[str] | None = None,
    include_linestyles: bool = True,
    include_linealpha: bool = False,
    include_dashes: bool = True,
) -> None:
    """Add the per-series style list arguments shared by the multi-series plotting commands."""
    group = arggroup(parser, "appearance")
    group.add_argument("-label", default=[], nargs="*", help="List of series label overrides")
    group.add_argument(
        "-color",
        "-colors",
        dest="color",
        type=color_arg,
        default=list(colordefault) if colordefault else [],
        nargs="*",
        help="List of line colors",
    )
    if include_linestyles:
        group.add_argument("-linestyle", default=[], nargs="*", help="List of line styles")
        group.add_argument("-linewidth", default=[], nargs="*", help="List of line widths")
    if include_linealpha:
        group.add_argument("-linealpha", default=[], nargs="*", help="List of line alphas (opacities)")
    if include_dashes:
        group.add_argument("-dashes", default=[], nargs="*", help="Dashes property of lines")


def addarg_figscale(
    parser: argparse.ArgumentParser, *, figscaledefault: float = 1.0, include_figwidthscale: bool = False
) -> None:
    """Add the figure size scale factor arguments."""
    group = arggroup(parser, "appearance")
    group.add_argument(
        "-figscale", type=float, default=figscaledefault, help="Scale factor for plot area. 1.0 is for single-column"
    )
    if include_figwidthscale:
        group.add_argument("-figwidthscale", type=float, default=1.0, help="Scale factor for plot width")


def addarg_filter(parser: argparse.ArgumentParser) -> None:
    """Add the spectrum smoothing filter arguments (get_filterfunc reads exactly these dests)."""
    group = arggroup(parser, "appearance")
    group.add_argument("-filtermovingavg", type=int, default=0, help="Smoothing length (1 is same as none)")
    group.add_argument(
        "-filtersavgol",
        nargs=2,
        help="Savitzky-Golay filter. Specify the window_length and poly_order, e.g. -filtersavgol 5 3",
    )


def addarg_action(parser: argparse.ArgumentParser, choices: Sequence[str], helptext: str) -> None:
    """Add the positional action argument that selects what the subcommand does."""
    parser.add_argument(
        "action",
        # optional so that main(argsraw=[], action=...) works, since parse_cli_args ignores
        # argsraw as soon as any keyword argument is given
        nargs="?",
        default=None,
        choices=choices,
        help=helptext,
    )


def suggest_names(name: str, candidates: "Collection[str]", *, count: int = 3) -> str:
    """Return a sentence that names the closest candidates, or an empty string when none is close.

    The sentence goes on the help line of an error, thus it carries no leading space. A name that
    differs only in case comes first, because that mistake is common and difflib scores a short name
    such as "te" against "Te" below its own threshold.
    """
    import difflib

    names = list(candidates)
    if samecase := [other for other in names if other.lower() == name.lower() and other != name]:
        return f"Did you mean {samecase[0]}?"

    matches = difflib.get_close_matches(name, names, n=count, cutoff=0.6)

    return f"Did you mean {', '.join(matches)}?" if matches else ""


def print_error(message: str, helptext: str = "") -> None:
    """Print an error to the standard error, then a help line that says what to do next.

    The error states the fault alone, thus a reader sees the remedy on its own line. rich colours the
    prefix in a terminal alone.
    """
    from rich.console import Console
    from rich.text import Text

    console = Console(stderr=True, highlight=False, soft_wrap=True)
    console.print(Text("error: ", style="bold red") + Text(message))
    if helptext:
        console.print(Text("help: ", style="bold cyan") + Text(helptext))


def print_warning(*values: object) -> None:
    """Print a warning to the standard error, thus --quiet keeps it and a script reads a clean product.

    rich colours the prefix in a terminal, and it writes plain text into a pipe or under NO_COLOR.
    """
    from rich.console import Console
    from rich.text import Text

    message = " ".join(str(value) for value in values)
    Console(stderr=True, highlight=False, soft_wrap=True).print(Text("WARNING: ", style="bold yellow") + Text(message))


def exit_with_error(message: str, helptext: str = "") -> t.NoReturn:
    """Print an error message and stop with a failing exit status.

    A mistake in the arguments earns a message rather than a traceback, and a script that runs the
    command sees that it failed. helptext names what the user can do next.
    """
    print_error(message, helptext)
    raise SystemExit(1)


def require_action(args: argparse.Namespace) -> None:
    """Stop with an error message when the caller gave no action."""
    if args.action is None:
        exit_with_error("no action was given", "Run with --help to see the available actions")


def addarg_show(parser: argparse.ArgumentParser) -> None:
    """Add --show, which opens the figure in a window before the save, and --open, which opens the file after it."""
    group = arggroup(parser, "output")
    group.add_argument("--show", action="store_true", help="Show the plot in a window before saving it")
    group.add_argument("--open", action="store_true", help="Open the saved file with its default application")


def addarg_quiet(parser: argparse.ArgumentParser) -> None:
    """Add the --quiet argument that hides the progress messages.

    A command writes its product with print_product, thus --quiet keeps that product and hides only the
    progress messages around it.
    """
    arggroup(parser, "output").add_argument(
        "--quiet", "-q", action="store_true", help="Hide the progress messages. Warnings and errors still appear"
    )


def addarg_verbose(parser: argparse.ArgumentParser) -> None:
    """Add the --verbose argument that shows the detail of each step.

    A command prints a summary of the work by default. --verbose adds the detail, e.g. the name of
    each file that the command reads, thus -v means the same on every command.
    """
    arggroup(parser, "output").add_argument(
        "--verbose", "-v", action="store_true", help="Show the detail of each step, e.g. each file that is read"
    )


def print_product(args: argparse.Namespace, *values: object) -> None:
    """Print the product of a command, which --quiet keeps.

    --quiet sends the progress messages to the null device. The product of the command, e.g. the table
    of --print_data or the listing of --listvariables, must reach the standard output even so. A script
    then reads that product with no progress message around it.
    """
    print(*values, file=getattr(args, "productstream", None) or sys.stdout)


def addarg_dpi(parser: argparse.ArgumentParser, *, default: int = 250) -> None:
    """Add the -dpi argument setting the resolution of a raster output file."""
    arggroup(parser, "output").add_argument("-dpi", type=int, default=default, help="Dots per inch for the output file")


def addarg_yscale(parser: argparse.ArgumentParser) -> None:
    """Add the -yscale argument that selects the scale of the vertical axis.

    "auto" leaves the choice to the command, which keeps --logscaley working. "lin" means "linear".
    """
    arggroup(parser, "appearance").add_argument(
        "-yscale",
        choices=["log", "linear", "lin", "auto"],
        default="auto",
        help="Scale of the vertical axis. auto lets the command choose",
    )


def resolve_yscale(args: argparse.Namespace) -> None:
    """Set args.logscaley from -yscale, which every plot helper reads.

    --logscaley is the older spelling of "-yscale log". Two arguments that ask for a different scale
    get a message rather than a silent precedence.
    """
    yscale = getattr(args, "yscale", "auto")
    if yscale == "auto":
        return

    wantlog = yscale == "log"
    if getattr(args, "logscaley", False) and not wantlog:
        exit_with_error(f"specify only one of --logscaley and -yscale {yscale}")

    args.logscaley = wantlog


def addarg_notitle(parser: argparse.ArgumentParser) -> None:
    """Add the --notitle argument that suppresses the plot title (set_plot_title reads this dest)."""
    arggroup(parser, "appearance").add_argument(
        "--notitle", action="store_true", help="Suppress the top title from the plot"
    )


def addarg_nolegend(parser: argparse.ArgumentParser) -> None:
    """Add the --nolegend argument that suppresses the plot legend."""
    arggroup(parser, "appearance").add_argument(
        "--nolegend", action="store_true", help="Suppress the legend from the plot"
    )


def addarg_maxpacketfiles(parser: argparse.ArgumentParser) -> None:
    """Add the -maxpacketfiles argument limiting how many packet files are read."""
    parser.add_argument(
        "-maxpacketfiles", "-maxpacketsfiles", type=int, default=None, help="Limit the number of packet files read"
    )


def check_time_selection(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argsraw: "Sequence[str] | None" = None,
    kwargs: "Collection[str] | None" = None,
) -> None:
    """Stop when the arguments name a time range in more than one way.

    get_time_range cannot make this test. A caller assigns the times that it returns back onto its own
    arguments, thus a second call for a second model path would read its own output as a second range.

    The test reads the arguments that the user wrote, because a value can be the same as the default of
    the parser: plottransitions gives -timestep a default of 70, and a user can also type that value.

    set_args_from_dict makes a keyword argument of the API into a default of the parser, thus a value
    that differs from the default counts, and so does a name that kwargs holds. A name that carries
    None counts for nothing, because a caller can forward an argument that it does not use.
    """
    argstrings = list(sys.argv[1:] if argsraw is None else argsraw)
    keywordnames = set(kwargs or ())
    flagsofdest = {
        action.dest: action.option_strings
        for action in parser._actions  # ruff:ignore[private-member-access]
    }
    allflags = list(itertools.chain.from_iterable(flagsofdest.values()))

    def givesflag(argstring: str, flag: str) -> bool:
        """Report whether the string of the command line gives the flag its value, as argparse reads it.

        argparse joins a value to a flag of one letter alone, thus -t300 gives -t the value 300, and
        -ts70 also reads as -t with the value s70. A string that names another flag, exactly or as a
        start of its name, belongs to that flag.
        """
        if argstring == flag or argstring.startswith(f"{flag}="):
            return True

        if len(flag) != 2 or flag.startswith("--") or not argstring.startswith(flag):
            return False

        base = argstring.partition("=")[0]

        return not any(other != flag and (base == other or other.startswith(base)) for other in allflags)

    def wasgiven(dest: str) -> bool:
        # a value of None selects no time, thus a caller that forwards an unused argument gives nothing
        if not hasattr(args, dest) or getattr(args, dest) is None:
            return False

        flags = flagsofdest.get(dest, [])
        # set_args_from_dict takes the dest of an argument or any of its option strings as a key
        if dest in keywordnames or any(flag.lstrip("-") in keywordnames for flag in flags):
            return True

        if any(givesflag(argstring, flag) for flag in flags for argstring in argstrings):
            return True

        return bool(getattr(args, dest) != parser.get_default(dest))

    given = [name for name in ("timestep", "timedays", "timemin", "timemax") if wasgiven(name)]
    # -timemin and -timemax bound one range, thus the pair counts as one way to give it. get_time_range
    # reads the range of -timedays alone, thus a bound beside it has no effect and must not pass
    ways = [f"-{name}" for name in ("timestep", "timedays") if name in given]
    if bounds := [f"-{name}" for name in ("timemin", "timemax") if name in given]:
        ways.append(" and ".join(bounds))

    if len(ways) > 1:
        exit_with_error(f"{', '.join(ways)} name the time range in more than one way", "Give only one of them")


def parse_cli_args(
    addargsfunc: Callable[[argparse.ArgumentParser], None],
    description: str | None,
    args: argparse.Namespace | None,
    argsraw: Sequence[str] | None = None,
    kwargs: dict[str, t.Any] | None = None,
) -> argparse.Namespace:
    """Return args unchanged if already parsed, otherwise parse the command line using the options defined by addargsfunc.

    Any keyword arguments override the parser defaults, and when at least one is given, the command line/argsraw is ignored.
    """
    if args is not None:
        return args

    import argcomplete

    parser = argparse.ArgumentParser(formatter_class=CustomArgHelpFormatter, description=description)
    addargsfunc(parser)
    # the dispatcher adds these to the parser that it builds, thus a direct call needs them here
    addarg_quiet(parser)
    addarg_collidingflags(parser)
    kwargs = kwargs or {}
    set_args_from_dict(parser, kwargs)
    argcomplete.autocomplete(parser)
    args = parser.parse_args([] if kwargs else argsraw)
    check_time_selection(parser, args, [] if kwargs else argsraw, kwargs)
    resolve_output_argument(args)

    return args


def resolve_outputfile(outputfile: Path | str | None, defaultoutputfile: Path | str) -> Path:
    """Return the output file path, appending the default filename if outputfile is unset or refers to a folder.

    A path with no file extension is treated as a folder and will be created if it does not exist.
    """
    if not outputfile:
        return Path(defaultoutputfile)

    outputfile = Path(outputfile)
    if outputfile.is_dir() or not outputfile.suffixes:
        outputfile.mkdir(parents=True, exist_ok=True)
        return outputfile / defaultoutputfile

    return outputfile


def resolve_frameset_paths(
    outputfile: Path | str | None, *, framecount: int, framename: str, productname: str | None = None
) -> tuple[Path, Path | None]:
    """Return the path template of one frame, and the path of the file that holds every frame.

    A run that draws several figures combines them into one product, e.g. a gif or a merged pdf.
    productname names that product. None says that the combining step names it, as merge_pdf_files
    takes the names of the first frame and the last one.

    A -o path that has a file extension names the product itself, thus the frames go in the folder that
    holds it. A -o path with no file extension names a folder. This makes that folder either way.
    """
    givenpath = Path(outputfile) if outputfile else Path()

    if productname is not None and givenpath.suffix and not givenpath.is_dir():
        # the folder of the product can carry a suffix of its own, e.g. results.v1, thus make it here
        # and let resolve_outputfile read it as a folder and not as the name of one frame
        givenpath.parent.mkdir(parents=True, exist_ok=True)

        return resolve_outputfile(givenpath.parent, framename), givenpath

    frametemplate = resolve_outputfile(outputfile, framename)
    if framecount > 1 and "{" not in frametemplate.name:
        msg = (
            f"'{frametemplate.name}' names one file, and this command writes {framecount} frames. Give "
            "a folder with -o, or a name that holds a field, e.g. -o 'frame_{timestep}.png'"
        )
        raise ValueError(msg)

    productpath = frametemplate.parent / productname if productname is not None else None

    return frametemplate, productpath


def set_args_from_dict(parser: argparse.ArgumentParser, kwargs: dict[str, t.Any]) -> None:
    """Set argparse defaults from a dictionary."""
    kwargs = kwargs.copy()  # keys are renamed to argument dests below, so don't mutate the caller's dict
    # set_defaults expects the dest of an argument. Here we allow the option strings to be used as keys
    for arg in parser._actions:  # ruff:ignore[private-member-access]
        for optstring in arg.option_strings:
            if optstring.lstrip("-") in kwargs and arg.dest not in kwargs:
                kwargs[arg.dest] = kwargs.pop(optstring.lstrip("-"))

    parser.set_defaults(**kwargs)
    # set required=False on all arguments to avoid errors about missing required arguments when we set defaults from kwargs
    for arg in parser._actions:  # ruff:ignore[private-member-access]
        if arg.default is not None:
            arg.required = False

    if unknown := {k: v for k, v in kwargs.items() if k not in (arg.dest for arg in parser._actions)}:  # ruff:ignore[private-member-access]
        msg = f"Unknown argument names: {unknown}"
        raise ValueError(msg)


def parse_range(rng: str, dictvars: dict[str, int]) -> Iterable[int]:
    """Parse a string with an integer range and return a list of numbers, replacing special variables in dictvars."""
    strparts = rng.split("-")

    if len(strparts) not in {1, 2}:
        msg = f"Bad range: '{rng}'"
        raise ValueError(msg)

    parts = [int(i) if i not in dictvars else dictvars[i] for i in strparts]
    start: int = parts[0]
    end: int = start if len(parts) == 1 else parts[1]

    if start > end:
        end, start = start, end

    return range(start, end + 1)


def parse_range_list(rngs: str | list[str] | list[int] | int, dictvars: dict[str, int] | None = None) -> list[int]:
    """Parse a string with comma-separated ranges or a list of range strings.

    Return a sorted list of integers in any of the ranges.
    """
    if isinstance(rngs, list):
        rngs = ",".join(str(x) for x in rngs)
    elif not isinstance(rngs, str):
        return [rngs]

    return sorted(set(itertools.chain.from_iterable([parse_range(rng, dictvars or {}) for rng in rngs.split(",")])))


def makelist(x: Sequence[t.Any] | str | Path | None) -> list[t.Any]:
    """If x is not a list (or is a string), make a list containing x."""
    if x is None:
        return []
    return [x] if isinstance(x, str | Path) else list(x)


def trim_or_pad(requiredlength: int, *listoflistin: t.Any) -> Sequence[Sequence[t.Any]]:
    """Make lists equal in length to requiredlength either by padding with None or truncating."""
    list_sequence = []
    for listin in listoflistin:
        listin_makelist = makelist(listin)

        listout = [listin_makelist[i] if i < len(listin_makelist) else None for i in range(requiredlength)]

        assert len(listout) == requiredlength
        list_sequence.append(listout)
    return list_sequence


def get_series_label(labels: Sequence[str | None], index: int, fallback: str) -> str:
    """Return the -label value for one series, or fallback when the user gave none for it.

    trim_or_pad pads the list with None, so an entry can be missing either as a None or, when the series
    count is not the model path count, by running off the end. An empty label is not a missing one:
    matplotlib gives no legend entry to a series labelled "", which is how one is left out of the legend.
    """
    label = labels[index] if 0 <= index < len(labels) else None

    return fallback if label is None else label


def flatten_list(listin: list[t.Any]) -> list[t.Any]:
    """Flatten a list of lists."""
    listout = []
    for elem in listin:
        if isinstance(elem, list):
            listout.extend(elem)
        else:
            listout.append(elem)
    return listout


def normalize_path_list(paths: PathArg, default: Path | str = ".") -> list[Path]:
    """Return a flat list of Paths from a scalar or (possibly nested) sequence of paths, using the default if none given."""
    if not paths:
        return [Path(default)]
    if isinstance(paths, str | Path):
        return [Path(paths)]
    return [Path(p) for p in flatten_list(list(paths))]


def get_filterfunc(args: argparse.Namespace) -> "Callable[[npt.ArrayLike], npt.NDArray[np.float64]] | None":
    """Use command line arguments to determine the appropriate filter function."""
    filterfunc = None
    dictargs = vars(args)

    if dictargs.get("filtermovingavg", False):

        def movavgfilterfunc(ylist: "npt.ArrayLike") -> "npt.NDArray[np.float64]":
            import numpy as np

            n = args.filtermovingavg
            arr_padded = np.pad(ylist, (n // 2, n - 1 - n // 2), mode="edge")
            return np.convolve(arr_padded, np.ones((n,)) / n, mode="valid")

        assert filterfunc is None
        filterfunc = movavgfilterfunc

    if dictargs.get("filtersavgol", False):
        from artistools.misc.general import savgol_filter

        window_length, polyorder = (int(x) for x in args.filtersavgol)

        def savgolfilterfunc(ylist: "npt.ArrayLike") -> "npt.NDArray[np.float64]":
            return savgol_filter(ylist, window_length=window_length, polyorder=polyorder)

        assert filterfunc is None
        filterfunc = savgolfilterfunc

        print("Applying Savitzky-Golay filter")

    return filterfunc
