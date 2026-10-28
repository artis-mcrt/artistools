"""Shared helpers for command-line argument parsing and list/path argument normalisation."""

import argparse
import copy
import itertools
import math
import operator
import sys
import typing as t
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from pathlib import Path

from artistools.misc.argparser import arggroup
from artistools.misc.argparser import CustomArgHelpFormatter
from artistools.misc.argparser import get_shown_flags
from artistools.misc.argparser import get_standard_flag
from artistools.misc.argparser import KeepGivenPaths
from artistools.misc.argparser import ReplaceGivenPaths
from artistools.misc.argparser import separate_trailing_folders
from artistools.misc.argparser import SuggestingArgumentParser
from artistools.misc.commandoutput import make_output_folder
from artistools.misc.commandoutput import OUTPUT_FORMATS
from artistools.misc.commandoutput import resolve_outputfile
from artistools.misc.console import exit_with_error
from artistools.misc.console import print_detail
from artistools.misc.console import print_heading
from artistools.misc.dirbins import get_dirbin_labels
from artistools.misc.dirbins import get_viewingdirectionbincount
from artistools.misc.general import flatten_list
from artistools.misc.general import format_range_list
from artistools.misc.general import parse_range_list
from artistools.misc.modelpaths import folder_is_artis_model
from artistools.misc.modelpaths import get_model_folder
from artistools.misc.modelpaths import path_is_codecomparison
from artistools.misc.remote import get_canonical_path
from artistools.misc.remote import is_remote_path
from artistools.misc.remote import model_path_from_text
from artistools.misc.seriesstyles import KeepDefaultColors
from artistools.misc.timesteps import get_time_range
from artistools.misc.units import time_days_arg
from artistools.misc.units import time_days_range_arg
from artistools.misc.units import TIME_UNIT_HELP
from artistools.misc.units import wavelength_angstroms_arg
from artistools.misc.units import WAVELENGTH_UNIT_HELP

if t.TYPE_CHECKING:
    from collections.abc import Collection

# a path argument arrives as a scalar, as a sequence, or as a nested sequence from repeated -modelpath
type PathArg = Path | str | Sequence[PathArg] | None


# an error message names the model folders that are near. A folder of models can hold many,
# thus this constant gives the maximum number of names
MAXNEARBYFOLDERS = 6


class CommaJoinAction(argparse.Action):
    """Join a repeated flag: "--ts 3 --ts 5" gives "3,5", which parse_range_list expands.

    Without this the last occurrence replaces the first, and the command drops a cell silently.
    """

    @t.override
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[t.Any] | None,
        option_string: str | None = None,
    ) -> None:
        """Put the joined text of every occurrence of this flag in the namespace."""
        previous = getattr(namespace, self.dest, None)
        text = ",".join(str(value) for value in values) if isinstance(values, list) else str(values)
        isfirstoccurrence = previous is self.default or previous is None
        setattr(namespace, self.dest, text if isfirstoccurrence else f"{previous},{text}")


class CellListAction(argparse.Action):
    """Store the cells of --modelgridindex as a sorted list, and add the cells of each repeated flag.

    "--cell 3 --cell 5-7" gives [3, 5, 6, 7]. Each command then reads a list, whatever text the user gave.
    """

    @t.override
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[t.Any] | None,
        option_string: str | None = None,
    ) -> None:
        """Put the cells of every occurrence of this flag in the namespace."""
        previous = getattr(namespace, self.dest, None)
        try:
            cells = get_cell_list(str(values))
        except ValueError as exc:
            # argparse names the flag in the message of an ArgumentError, and it stops with no traceback
            msg = f"'{values}' names no cells. Give a cell, a range such as 3-7, or a list such as 1,4,9"
            raise argparse.ArgumentError(self, msg) from exc
        isfirstoccurrence = previous is self.default or previous is None
        setattr(namespace, self.dest, cells if isfirstoccurrence else sorted({*previous, *cells}))


def get_cell_list(cells: "str | t.SupportsIndex | Iterable[str | t.SupportsIndex]") -> list[int]:
    """Return the sorted integers of a text of ranges, an integer, or a sequence, e.g. "3-7,9", 12, or [4, "6-7"].

    The values of --modelgridindex and --ion-stages use this function. A keyword argument of main() does not pass
    through the parser, thus the function also takes an integer or a list. A float gives a TypeError.
    """
    if isinstance(cells, str):
        return parse_range_list(cells)
    if isinstance(cells, Iterable):
        return sorted({cell for item in cells for cell in get_cell_list(item)})
    return [operator.index(cells)]


def addarg_viewingangle(parser: argparse.ArgumentParser, allow_select_all: bool = False) -> None:
    """Add the viewing direction selection and averaging arguments shared by the plotting commands."""
    arggroup(parser, "Viewing direction").add_argument(
        "-plotvspecpol",
        type=int,
        metavar="n",
        nargs="+",
        help="Plot viewing angles from vspecpol virtual packets. Expects int for angle = spec number in vspecpol files",
    )

    arggroup(parser, "Viewing direction").add_argument(
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
    parser.set_defaults(allowalldirbins=allow_select_all)

    arggroup(parser, "Viewing direction").add_argument(
        "--usedegrees",
        action="store_true",
        help="Show the angles of the viewing directions in degrees, and not as cos θ and radians",
    )

    # averaging over one angle leaves one bin per index of the other, so the two cannot be combined. argparse
    # enforces this once here for every command that takes these flags
    averagegroup = arggroup(parser, "Viewing direction").add_mutually_exclusive_group()

    averagegroup.add_argument(
        "--average_over_phi_angle",
        action="store_true",
        help="Average over phi (azimuthal) viewing angles to make direction bins into polar angle bins",
    )

    # the hidden alias keeps the old spelling. argparse writes a warning when a command gives it
    averagegroup.add_argument(
        "--average_every_tenth_viewing_angle",
        dest="average_over_phi_angle",
        action="store_true",
        deprecated=True,
        help=argparse.SUPPRESS,
    )

    averagegroup.add_argument(
        "--average_over_theta_angle",
        action="store_true",
        help="Average over theta (polar) viewing angles to make direction bins into azimuthal angle bins",
    )


def addarg_pathoption(parser: argparse.ArgumentParser, flag: str, dest: str, *, multiplepaths: bool) -> None:
    """Accept an option that names the same paths as a positional argument.

    Some commands take the paths as a positional argument and others take --modelpath. A user who learns
    one form must not meet "unrecognized arguments" with the other, thus the option stands beside the
    positional. It stays out of the help text, because the positional already gives the paths a name.
    """
    optionkwargs: dict[str, t.Any] = {
        "dest": dest,
        "action": ReplaceGivenPaths,
        "type": model_path_from_text,
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
    """Add the ARTIS model path argument (--modelpath option, or a positional path when positional=True)."""
    kwargs: dict[str, t.Any] = {"type": model_path_from_text, "default": default, "help": helptext}
    if multiplepaths:
        kwargs["nargs"] = "*"
    if positional:
        # a positional argument with no nargs is required, and the option form could then never replace it
        parser.add_argument("modelpath", action=KeepGivenPaths, nargs=kwargs.pop("nargs", "?"), **kwargs)
        addarg_pathoption(parser, "-modelpath", "modelpath", multiplepaths=multiplepaths)
    else:
        if required:
            kwargs["required"] = True
        parser.add_argument("-modelpath", **kwargs)


def item_names_a_path(item: str) -> bool:
    """Return whether a positional argument names a path in any position.

    A path that a user writes has a parent folder that exists, e.g. "models/mymodel", or it names a
    virtual codecomparison data set. A separator alone does not make a path, because an estimator
    variable can have one, e.g. heating_dep/total_dep.
    """
    if not item:
        return False

    parent = Path(item).parent

    return (parent != Path() and parent.is_dir()) or path_is_codecomparison(item)


def item_names_a_folder(item: str) -> bool:
    """Return whether the last positional argument names the ARTIS folder and not a name of the command.

    A folder that exists names the model, even when the command has an item of the same name. A name
    that is not last stays an item. A variable thus keeps its meaning when a folder has the same name.
    """
    # a remote path has no "=" in its host, thus a style value with a colon, e.g. Te@color=tab:red, names no folder
    return bool(item) and (is_remote_path(item) or Path(item).is_dir() or item_names_a_path(item))


def addarg_positional_items(
    parser: argparse.ArgumentParser, *, dest: str, metavar: str, helptext: str
) -> argparse.Action:
    """Add the positional arguments of a command that reads the ARTIS folder as the last one.

    A command that names its items on the command line, e.g. the estimator variables of
    plotestimators, adds them with this function. resolve_positional_modelpath then removes the ARTIS
    folder from the end. The caller can put an argcomplete completer on the action that this function
    returns.
    """
    # a user can write a flag between two items, thus this parser reads the positional arguments intermixed
    if isinstance(parser, SuggestingArgumentParser):
        parser.wantsintermixed = True

    return parser.add_argument(dest, nargs="*", default=[], metavar=metavar, help=helptext)


def resolve_positional_modelpath(args: argparse.Namespace, dest: str) -> list[str]:
    """Remove the ARTIS folder from the end of the positional arguments, and return the other items.

    The ARTIS folder comes last, thus only the last argument can name a folder. A folder in an earlier
    place gives an error, because a name in that place is a name of the command. This one rule serves
    every command that takes positional items. Thus every such command has the same order.

    The command adds --modelpath with the default of addarg_modelpath, which is None. A value that is
    not None then shows that the user gave --modelpath. The working folder applies at the end.
    """
    items: list[str] = list(getattr(args, dest))

    if items and item_names_a_folder(items[-1]):
        givenpath = get_canonical_path(model_path_from_text(items.pop()))
        # the default of such a command is None, thus a different value is one that the user wrote.
        # A command can also take many paths, and then only the positional argument names the model
        given_modelpath = getattr(args, "modelpath", None)
        if isinstance(given_modelpath, str | Path) and Path(given_modelpath) != givenpath:
            exit_with_error(
                f"The folder '{givenpath}' and --modelpath '{given_modelpath}' name two different models",
                "Give the folder one time",
            )
        args.modelpath = givenpath

    # a bare name that holds an ARTIS model is a folder that the user wrote too early. A name that only
    # matches a folder stays an item, thus a variable keeps its meaning when a folder has that name
    def item_is_misplaced(item: str) -> bool:
        return item_names_a_path(item) or folder_is_artis_model(item)

    if misplaced := [item for item in items if item_is_misplaced(item)]:
        kept = [item for item in items if not item_is_misplaced(item)]
        example = " ".join([*kept, *misplaced, str(getattr(args, "modelpath", "") or "")]).strip()
        exit_with_error(
            f"'{misplaced[0]}' names a folder, and the ARTIS folder comes after the other arguments",
            f"Write the folder last, e.g. {example}",
        )

    if getattr(args, "modelpath", "") is None:
        args.modelpath = Path()

    setattr(args, dest, items)

    return items


def artis_subfolders(folder: Path) -> list[str]:
    """Return the names of the subfolders of this folder that hold an ARTIS model.

    A user often runs a command one level above the model, e.g. in the folder that holds every model.
    An error message can then name the folders that are near. The user does not have to list the folder.
    """
    try:
        children = sorted(child.name for child in folder.iterdir() if folder_is_artis_model(child))
    except OSError:
        # the folder can give a permission error. A message about the model helps more than that error
        return []

    return children[:MAXNEARBYFOLDERS]


def addarg_output(
    parser: argparse.ArgumentParser,
    *,
    kind: t.Literal["file", "folder"],
    defaultname: str | None = None,
    default: t.Any = None,
    helptext: str | None = None,
) -> None:
    """Add the --outputfile/-o argument, and record what the command writes.

    A command writes one file or a folder of files, and kind says which. resolve_output_argument reads
    that word after the parse: it gives a file the defaultname of the command when -o names a folder,
    and it makes the folder that -o names either way. Thus every command keeps the promise of the help
    text, and no command writes that rule again.

    A command that names its own frames takes no defaultname, because resolve_frameset_paths gives each
    frame a name of its own.
    """
    if kind == "file":
        defaulthelptext = "Path/filename for the output file"
        rule = "A path with no file extension names a folder, which the command creates"
    else:
        defaulthelptext = "Path for the output files"
        rule = "The command creates this folder"
    # -o is a Path on every command, thus no command reads a text where another reads a Path
    kwargs: dict[str, t.Any] = {
        "dest": "outputfile",
        "default": default,
        "type": Path,
        "help": f"{helptext or defaulthelptext}. {rule}",
    }

    arggroup(parser, "Output").add_argument("-outputfile", "-outputpath", "-o", **kwargs)
    parser.set_defaults(outputkind=kind, outputdefaultname=defaultname)


def resolve_output_argument(args: argparse.Namespace) -> None:
    """Apply the rule of -o that addarg_output recorded on the parser of the command.

    A command that writes one file takes the name of that file when -o names a folder. A command that
    writes a folder of files gets that folder. The folder exists after this either way.
    """
    kind = getattr(args, "outputkind", None)
    if kind is None:
        return

    outputfile = getattr(args, "outputfile", None)
    if kind == "folder":
        # a command that takes no -o names its own folder, thus there is nothing to make
        if outputfile:
            make_output_folder(outputfile, "writes")
    elif (defaultname := getattr(args, "outputdefaultname", None)) is not None:
        # resolve_outputfile gives the name of the command when -o names a folder or nothing
        args.outputfile = resolve_outputfile(outputfile, defaultname)


def addarg_modelgridindex(
    parser: argparse.ArgumentParser, *, default: list[int] | None = None, helptext: str | None = None
) -> None:
    """Add the --modelgridindex/--cell/--mgi argument that selects the model grid cell or cells.

    Every command reads the same text: a number, or a range such as 3-7, or a list such as 1,4,9.
    CellListAction stores the sorted list of the cells, and get_single_modelgridindex gives the one
    cell that a command which reads one cell needs.
    """
    arggroup(parser, "Cell selection").add_argument(
        "-modelgridindex",
        "-cell",
        "-mgi",
        action=CellListAction,
        default=default,
        help=helptext or "Model grid cell to plot, e.g. 12 or a range 3-7",
    )


def get_single_modelgridindex(cells: Sequence[int] | None) -> int | None:
    """Return the one cell of the cells that --modelgridindex names, or None when it names none.

    Every command reads the same text, thus a range reaches a command that plots one cell. Such a
    range earns a message that says so, in place of a cell that the text does not name.
    """
    if not cells:
        return None

    if len(cells) > 1:
        msg = (
            f"--modelgridindex '{format_range_list(cells)}' names {len(cells)} cells, and this command reads one. "
            "Give one cell, e.g. --cell 12"
        )
        raise ValueError(msg)

    return cells[0]


class UnsupportedArgument(argparse.Action):
    """Stop the command, and name the argument to give in place of the one that the user gave."""

    def __init__(self, option_strings: "Sequence[str]", dest: str, instead: str, **kwargs: t.Any) -> None:
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
        assert isinstance(parser, SuggestingArgumentParser), "every parser of a command is this class"
        parser.exit_with_help(f"{option_string} is not an argument of this command", f"Give {self.instead} instead")


def addarg_unsupported(parser: argparse.ArgumentParser, *flags: str, instead: str) -> None:
    """Declare an argument that this command does not take, so that a user gets a clear message.

    argparse joins a value to a single-dash flag, thus "-timestep 30" on a parser that declares -t but
    no -timestep reads as "-t imestep". A declared name gives a message that names the right argument.
    """
    parser.add_argument(*flags, action=UnsupportedArgument, instead=instead, default=argparse.SUPPRESS)


def addarg_timestep(parser: argparse.ArgumentParser, *, default: t.Any = None, helptext: str | None = None) -> None:
    """Add the --timestep/--ts argument that selects the timestep or the timesteps.

    Every command reads the same text: a number, a range such as 45-65, or "last". get_time_range
    refuses a list such as 4,9, because a plot reads one range of timesteps. get_single_timestep
    gives the one timestep that a command which plots one timestep needs.
    """
    arggroup(parser, "Time selection").add_argument(
        "-timestep",
        "-ts",
        action=CommaJoinAction,
        default=default,
        help=helptext or "Timestep to plot, e.g. 40, a range 45-65, or last",
    )


def addarg_plotinvalidpart(parser: argparse.ArgumentParser, plotted: str) -> None:
    """Add --plotinvalidpart, which also takes the times outside the valid range of the arrival times.

    plotted names what the command then plots, e.g. "the spectra".
    """
    parser.add_argument(
        "--plotinvalidpart",
        action="store_true",
        help=(
            f"Plot {plotted} also outside the valid range of the arrival times. At these times, the light from a part"
            " of the ejecta does not reach the observer, because of the light travel time"
        ),
    )


def addarg_averageoverphi(parser: argparse.ArgumentParser, optionname: str = "-dirbin") -> None:
    """Add --average-over-phi-angle, which makes a direction bin of optionname into the mean over its phi bins.

    The option has the meaning of --average-over-phi-angle of plotspectra. With the option, --dirbin 90 takes the
    costheta bin 0.8 to 1. Without it, --dirbin 90 takes the one direction bin 90.
    """
    optionname = get_standard_flag(optionname)
    parser.add_argument(
        "--average_over_phi_angle",
        action="store_true",
        help=(
            f"Take the mean over the phi bins of the costheta bin of {optionname}. Then {optionname} must be the first"
            " bin of its costheta bin, e.g. 90 for the pole at +z"
        ),
    )


def addarg_wavelengthrange(parser: argparse.ArgumentParser, selected: str) -> None:
    """Add --lambdamin and --lambdamax (also --xmin and --xmax), which select a range of wavelength in Angstroms.

    selected names what the range selects, e.g. "the lines". A plot command with a wavelength axis takes
    addarg_axislimits, because there --xmin and --xmax are the limits of the axis.
    """
    parser.add_argument(
        "-xmin",
        "-lambdamin",
        type=wavelength_angstroms_arg,
        help=f"Minimum wavelength of {selected}. {WAVELENGTH_UNIT_HELP}",
    )
    parser.add_argument(
        "-xmax",
        "-lambdamax",
        type=wavelength_angstroms_arg,
        help=f"Maximum wavelength of {selected}. {WAVELENGTH_UNIT_HELP}",
    )
    parser.set_defaults(xrangeisselection=True)


def addarg_sliceaxis(parser: argparse.ArgumentParser, *, default: str, helptext: str) -> None:
    """Add --axis, which names a half-axis of a 3D model, e.g. +z or -x. A bare axis name, e.g. z, gives the + side."""
    parser.add_argument(
        "-axis",
        default=default,
        choices=["x", "y", "z", "+x", "-x", "+y", "-y", "+z", "-z"],
        help=f"{helptext}. Hint: for a negative half-axis use e.g. --axis=-z",
    )


def resolve_slice_axis(args: argparse.Namespace) -> None:
    """Set args.sliceaxis to the axis name of --axis, and args.positive_axis to whether it names the + side."""
    args.positive_axis = args.axis[0] != "-"
    args.sliceaxis = args.axis.lstrip("+-")


def addarg_timedays(
    parser: argparse.ArgumentParser,
    *,
    kind: t.Literal["rangestr", "str", "float"] = "rangestr",
    helptext: str | None = None,
) -> None:
    """Add the --timedays/--time/-t argument, either as a range string like 50-100 or a single value.

    -t means --timedays on every command, thus a user needs no knowledge of which other arguments a
    command takes. A command that takes no -timestep calls addarg_unsupported for that name, because
    argparse joins a value to a single-dash flag and would read "-timestep 30" as "-t imestep".
    """
    group = arggroup(parser, "Time selection")
    flags = ("-timedays", "-time", "-t")
    if kind == "rangestr":
        group.add_argument(
            *flags,
            dest="timedays",
            nargs="?",
            type=time_days_range_arg,
            help=f"{helptext or 'Range of times to plot (e.g. 50-100)'}. {TIME_UNIT_HELP}",
        )
    elif kind == "float":
        group.add_argument(*flags, type=time_days_arg, help=f"{helptext or 'Time to plot'}. {TIME_UNIT_HELP}")
    else:
        group.add_argument(*flags, type=time_days_range_arg, help=f"{helptext or 'Time to plot'}. {TIME_UNIT_HELP}")


def addarg_timeminmax(
    parser: argparse.ArgumentParser, *, helptext_min: str = "Lower time", helptext_max: str = "Upper time"
) -> None:
    """Add the --timemin and --timemax arguments bounding a time range in days."""
    group = arggroup(parser, "Time selection")
    group.add_argument("-timemin", type=time_days_arg, help=f"{helptext_min}. {TIME_UNIT_HELP}")
    group.add_argument("-timemax", type=time_days_arg, help=f"{helptext_max}. {TIME_UNIT_HELP}")


def addarg_axislimits(
    parser: argparse.ArgumentParser,
    *,
    xmindefault: float | None = None,
    xmaxdefault: float | None = None,
    xminhelp: str = "Plot range: minimum x value",
    xmaxhelp: str = "Plot range: maximum x value",
    include_x: bool = True,
    include_y: bool = True,
    wavelength_aliases: bool = False,
    xtype: Callable[[str], float] | None = None,
) -> None:
    """Add the --xmin/--xmax and --ymin/--ymax plot range arguments.

    A command whose x axis is a wavelength takes wavelength_aliases, which adds the --lambdamin and --lambdamax
    spellings of the same arguments. Its x axis is in Angstroms, thus a value can also give a different unit, e.g.
    600nm. plotspectra has its own type, because the unit of its axis comes from --xunit.
    """
    group = arggroup(parser, "Appearance")
    if include_x:
        xminflags = ("-xmin", "-lambdamin") if wavelength_aliases else ("-xmin",)
        xmaxflags = ("-xmax", "-lambdamax") if wavelength_aliases else ("-xmax",)
        if xtype is None:
            xtype = wavelength_angstroms_arg if wavelength_aliases else float
        if xtype is wavelength_angstroms_arg:
            xminhelp = f"{xminhelp}. {WAVELENGTH_UNIT_HELP}"
            xmaxhelp = f"{xmaxhelp}. {WAVELENGTH_UNIT_HELP}"
        group.add_argument(*xminflags, dest="xmin", type=xtype, default=xmindefault, help=xminhelp)
        group.add_argument(*xmaxflags, dest="xmax", type=xtype, default=xmaxdefault, help=xmaxhelp)
    if include_y:
        group.add_argument("-ymin", type=float, default=None, help="Plot range: y-axis minimum")
        group.add_argument("-ymax", type=float, default=None, help="Plot range: y-axis maximum")


def positive_int_arg(text: str) -> int:
    """Return the integer of the text, and reject a value below 1 when argparse reads the command line.

    A count of 0 or below passed argparse, and the command then failed after it read all the data.
    """
    try:
        value = int(text)
    except ValueError as exc:
        msg = f"invalid int value: {text!r}"
        raise argparse.ArgumentTypeError(msg) from exc
    if value < 1:
        msg = f"{value} is not a positive count. Give 1 or more"
        raise argparse.ArgumentTypeError(msg)
    return value


def positive_float_arg(text: str) -> float:
    """Return the number of the text, and reject a value that is not finite and above 0 when argparse reads it.

    A factor such as --linewidthscale -1 passed argparse, and matplotlib then failed after the command read all the
    data.
    """
    try:
        value = float(text)
    except ValueError as exc:
        msg = f"invalid float value: {text!r}"
        raise argparse.ArgumentTypeError(msg) from exc
    if not math.isfinite(value) or value <= 0.0:
        msg = f"{text} is not a finite number above 0"
        raise argparse.ArgumentTypeError(msg)
    return value


def addarg_figscale(
    parser: argparse.ArgumentParser, *, include_figwidthscale: bool = False, helptext: str | None = None
) -> None:
    """Add the figure size scale factor arguments."""
    group = arggroup(parser, "Appearance")
    group.add_argument(
        "-figscale",
        type=float,
        default=1.0,
        help=helptext or "Scale factor for plot area. 1.0 fills the text width of a page",
    )
    if include_figwidthscale:
        group.add_argument("-figwidthscale", type=float, default=1.0, help="Scale factor for plot width")


def addarg_filter(parser: argparse.ArgumentParser) -> None:
    """Add the spectrum smoothing filter arguments (get_filterfunc reads exactly these dests)."""
    group = arggroup(parser, "Appearance")
    group.add_argument("-filtermovingavg", type=int, default=0, help="Smoothing length (1 is same as none)")
    group.add_argument(
        "-filtersavgol",
        nargs=2,
        help="Savitzky-Golay filter. Specify the window_length and poly_order, e.g. --filtersavgol 5 3",
    )


def addarg_action(parser: argparse.ArgumentParser, choices: Sequence[str], helptext: str) -> None:
    """Add the positional action argument that selects what the subcommand does."""
    parser.add_argument(
        "action",
        # optional so that main(argsraw=[], action=...) works, because the keyword gives the default
        # and the command line then holds no action
        nargs="?",
        default=None,
        choices=choices,
        help=helptext,
    )


def require_action(args: argparse.Namespace) -> None:
    """Stop with an error message when the caller gave no action.

    The help line names each action and the command that shows the help. The dispatcher puts the parser of the
    command into args.argparser. A direct call of main has no such parser.
    """
    if args.action is not None:
        return

    parser = getattr(args, "argparser", None)
    choices = next(
        (action.choices for action in getattr(parser, "_actions", []) if action.dest == "action" and action.choices), ()
    )
    givetext = f"Give one of the actions {', '.join(map(str, choices))}. " if choices else ""
    helptext = f"Run `{parser.prog} --help`" if parser is not None else "Run the command with --help"
    exit_with_error("The command needs an action", f"{givetext}{helptext} to see what each action does")


def addarg_residuals(parser: argparse.ArgumentParser) -> None:
    """Add the options for the residual panel and its baseline series."""
    if isinstance(parser, SuggestingArgumentParser):
        parser.wantsintermixed = True
    group = arggroup(parser, "Appearance")
    baselinehelp = (
        "Select the baseline series for the residual panel in plot order, with the first series at index 0."
        " Without INDEX, use series 0. The plot must have one frame"
    )
    for flag, helptext in [
        ("-residualbaselineseries", baselinehelp),
        ("-residuals", argparse.SUPPRESS),
        ("-res", argparse.SUPPRESS),
    ]:
        group.add_argument(
            flag,
            dest="residualbaselineseries",
            type=int,
            nargs="?",
            const=0,
            default=None,
            metavar="INDEX",
            help=helptext,
        )
    # The old flag takes no value, because a numeric model path must stay positional.
    group.add_argument(
        "--residuals", dest="residualbaselineseries", action="store_const", const=0, help=argparse.SUPPRESS
    )
    group.add_argument(
        "-residual",
        dest="residuals",
        type=int,
        nargs="*",
        default=None,
        metavar="INDEX",
        help=(
            "Add a residual panel for the selected series."
            " By default, show series / baseline for flux and series minus baseline for magnitudes."
            " Give indices in plot order, with the first series at index 0."
            " By default, include all series except the baseline."
            " Print the root mean square (RMS) of each residual."
            " --write-data also writes this number"
        ),
    )
    group.add_argument(
        "-residualtype",
        choices=["absolute", "relative", "relativelog"],
        default="relative",
        help=(
            "Select the residual type: absolute shows series minus baseline."
            " Relative shows series / baseline on a linear y axis."
            " Relativelog shows the same ratio on a logarithmic y axis."
            " The default is relative. Magnitude panels always show series minus baseline on a linear y axis."
            " The statistics use series minus baseline for every type"
        ),
    )
    group.add_argument(
        "-residualymax",
        type=float,
        default=None,
        help=(
            "Set the maximum y value of the residual panel."
            " By default, use the data range. The magnitude axis shows this value at the bottom"
        ),
    )


def addarg_show(parser: argparse.ArgumentParser) -> None:
    """Add --show, which opens the figure in a window before the save, and --open, which opens the file after it."""
    group = arggroup(parser, "Output")
    group.add_argument("--show", action="store_true", help="Show the plot in a window before saving it")
    group.add_argument("--open", action="store_true", help="Open the saved file with its default application")


def addarg_darkmode(parser: argparse.ArgumentParser) -> None:
    """Add --darkmode, which gives a plot white text, frames, and ticks for a dark background.

    Each command that writes a plot adds it, and save_figure applies it.
    """
    arggroup(parser, "Output").add_argument(
        "--darkmode",
        action="store_true",
        help=(
            "Use white text, frames, and ticks for a dark background."
            " A PDF or SVG file has a transparent background, and a file in a different format has a black background"
        ),
    )


def addarg_quiet(parser: argparse.ArgumentParser) -> None:
    """Add the --quiet argument that hides the progress messages.

    A command writes its product with print_product, thus --quiet keeps that product and hides only the
    progress messages around it.
    """
    arggroup(parser, "Output").add_argument(
        "--quiet", "-q", action="store_true", help="Hide the progress messages. Warnings and errors still appear"
    )


def addarg_force(parser: argparse.ArgumentParser) -> None:
    """Add the -f and --force arguments, which permit a destructive action with no prompt, as rm -f does."""
    arggroup(parser, "Output").add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Do destructive actions with no prompt, e.g. overwrite the output files that exist",
    )


def addarg_verbose(parser: argparse.ArgumentParser) -> None:
    """Add the --verbose argument that shows the detail of each step.

    A command prints a summary of the work by default. --verbose adds the detail, e.g. the name of
    each file that the command reads.

    -v is the short form, but some commands gave -v to --velocity or to --rhoscale before this
    argument existed. A script holds such a command, thus -v keeps that meaning there as a hidden alias, and
    --verbose is the only form. Declare the argument of the command first, then call this function.
    """
    flags = ["--verbose"] if "-v" in parser._option_string_actions else ["--verbose", "-v"]  # ruff:ignore[private-member-access]
    arggroup(parser, "Output").add_argument(
        *flags, action="store_true", help="Show the detail of each step, e.g. each file that is read"
    )


def addarg_output_format(parser: argparse.ArgumentParser) -> None:
    """Add --output-format, which selects the format of a table, as --output-format of ruff does."""
    arggroup(parser, "Output").add_argument(
        "--output-format",
        choices=OUTPUT_FORMATS,
        default=None,
        help=(
            "The format of a table on the standard output or in a file. The default is text, or the format that the"
            " suffix of the file gives, e.g. .json"
        ),
    )


def addarg_table_output(parser: argparse.ArgumentParser) -> None:
    """Add -o and -f to a command that prints a table.

    The command can then write the table to a file in place of the standard output.
    """
    addarg_output(
        parser,
        kind="file",
        helptext=(
            "Write the table to this file in place of the standard output. The suffix .json, .csv, or .parquet gives"
            " that format, and another suffix gives text"
        ),
    )
    addarg_force(parser)
    addarg_output_format(parser)


def addarg_data_output(parser: argparse.ArgumentParser) -> None:
    """Add the --print-data and --write-data arguments, which give the plotted data as text."""
    group = arggroup(parser, "Output")
    group.add_argument(
        "--print_data",
        action="store_true",
        help="Print the plotted data to the standard output, with a space between the values",
    )
    group.add_argument(
        "--write_data",
        action="store_true",
        help="Write the plotted data to a .txt file beside the figure, with a space between the values",
    )
    addarg_output_format(parser)


def addarg_dpi(parser: argparse.ArgumentParser, *, default: int = 250) -> None:
    """Add the --dpi argument setting the resolution of a raster output file."""
    arggroup(parser, "Output").add_argument("-dpi", type=int, default=default, help="Dots per inch for the output file")


def addarg_labelfontsize(parser: argparse.ArgumentParser) -> None:
    """Add the --labelfontsize argument that sets the font size of the tick labels and the axis labels."""
    arggroup(parser, "Appearance").add_argument(
        "-labelfontsize",
        type=float,
        default=None,
        help="Font size of the tick labels and the axis labels. The default comes from the artistools matplotlibrc",
    )


def addarg_yscale(parser: argparse.ArgumentParser, *, default: str = "auto") -> None:
    """Add the --yscale argument that selects the scale of the vertical axis.

    "auto" reads the drawn values and takes a log scale when they cover more than one order of
    magnitude. It keeps --logscaley working, which asks for a log scale whatever the values are.
    "lin" means "linear".

    The argument has no default, thus resolve_yscale can find an explicit --yscale that gives a different
    scale from --logscaley. args.defaultyscale holds the default of the command.
    """
    arggroup(parser, "Appearance").add_argument(
        "-yscale",
        choices=["log", "linear", "lin", "auto"],
        default=None,
        help=(
            "Scale of the vertical axis. auto takes a log scale for values that cover a wide range."
            f" The default is {default}"
        ),
    )
    # the older spelling of "-yscale log"
    parser.add_argument("--logscaley", action="store_true", help=argparse.SUPPRESS)
    parser.set_defaults(defaultyscale=default)


def resolve_yscale(args: argparse.Namespace) -> None:
    """Set args.yscale to the scale that the plot takes, and set args.logscaley from it.

    --logscaley is the older spelling of "--yscale log", thus it replaces the default of the command.
    If two arguments ask for different scales, the command stops with an error message. The result
    has the spelling "linear" for "lin", thus the code that reads args.yscale compares with one spelling.
    """
    if not hasattr(args, "yscale"):
        return

    logscaley = getattr(args, "logscaley", False)
    yscale = args.yscale
    if logscaley and yscale in {"linear", "lin"}:
        exit_with_error(f"Specify only one of --logscaley and --yscale {yscale}")
    # "auto" chooses a scale, thus --logscaley gives that choice
    if yscale is None or (logscaley and yscale == "auto"):
        yscale = "log" if logscaley else getattr(args, "defaultyscale", "auto")

    args.yscale = "linear" if yscale == "lin" else yscale
    if args.yscale != "auto":
        args.logscaley = args.yscale == "log"


def addarg_notitle(parser: argparse.ArgumentParser) -> None:
    """Add the --notitle argument that suppresses the plot title (set_plot_title reads this dest)."""
    arggroup(parser, "Appearance").add_argument(
        "--notitle", action="store_true", help="Suppress the top title from the plot"
    )


def addarg_legend(parser: argparse.ArgumentParser) -> None:
    """Add the arguments of the legend: --nolegend, --legendframe, --legend-above, and --legendcols."""
    arggroup(parser, "Appearance").add_argument(
        "--nolegend", action="store_true", help="Suppress the legend from the plot"
    )
    arggroup(parser, "Appearance").add_argument(
        "-legendcols",
        type=positive_int_arg,
        default=None,
        help=(
            "The number of columns of the legend. The default gives a long legend more columns, thus the legend takes"
            " half of the frame height at most"
        ),
    )
    arggroup(parser, "Appearance").add_argument(
        "--legendframe",
        action="store_true",
        help="Draw the legend on a white background. The legend text then stays readable over the data",
    )
    arggroup(parser, "Appearance").add_argument(
        "--legend-above",
        action="store_true",
        help="Put the legend above the frame. The data then fill the frame, and the legend takes no room on the y axis",
    )


def check_time_selection(
    parser: SuggestingArgumentParser,
    args: argparse.Namespace,
    argsraw: "Sequence[str] | None" = None,
    kwargs: "Collection[str] | None" = None,
) -> None:
    """Stop when the arguments name a time range in more than one way.

    get_time_range cannot make this test. A caller assigns the times that it returns back onto its own
    arguments, thus a second call for a second model path would read its own output as a second range.

    The test reads the arguments that the user wrote, because a value can be the same as the default of
    the parser: a command can give --timestep a default of "last", and a user can also type that value.

    set_args_from_dict makes a keyword argument of the API into a default of the parser, thus a value
    that differs from the default counts, and so does a name that kwargs holds. A name that carries
    None counts for nothing, because a caller can forward an argument that it does not use.
    """
    # split a joined value such as -ts70 here as well, or the scan below reads it as -t
    argstrings = parser.split_joined_flags(list(sys.argv[1:] if argsraw is None else argsraw))
    keywordnames = set(kwargs or ())
    flagsofdest = {
        action.dest: action.option_strings
        for action in parser._actions  # ruff:ignore[private-member-access]
    }
    allflags = list(itertools.chain.from_iterable(flagsofdest.values()))

    def givesflag(argstring: str, flag: str) -> bool:
        """Report whether the string of the command line gives the flag its value, as argparse reads it.

        The split above separates a value from a flag of more than one letter, thus only a flag of
        one letter still carries a joined value here, e.g. -t300. A string that names another flag,
        exactly or as a start of its name, belongs to that flag.
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
    ways = [f"--{name}" for name in ("timestep", "timedays") if name in given]
    if bounds := [f"--{name}" for name in ("timemin", "timemax") if name in given]:
        ways.append(" and ".join(bounds))

    if len(ways) > 1:
        exit_with_error(f"{', '.join(ways)} name the time range in more than one way", "Give only one of them")


def check_range_order(args: argparse.Namespace) -> None:
    """Stop when a range of the selection has a lower end above its upper end.

    Such a range selects no data, thus the command gave an empty plot or an error after a long read. Two equal ends
    select one time, e.g. for decayproducts. The limits of a plot axis can have the opposite order, e.g. for a
    magnitude, thus this function checks only a range of the selection.
    """
    bounds = [("timemin", "timemax", "--timemin", "--timemax")]
    if getattr(args, "xrangeisselection", False):
        bounds.append(("xmin", "xmax", "--lambdamin", "--lambdamax"))
    for lowerdest, upperdest, lowerflag, upperflag in bounds:
        lower, upper = getattr(args, lowerdest, None), getattr(args, upperdest, None)
        if isinstance(lower, float | int) and isinstance(upper, float | int) and lower > upper:
            exit_with_error(
                f"{lowerflag} {lower:g} is above {upperflag} {upper:g}, thus the range selects nothing",
                f"Give the lower end to {lowerflag} and the upper end to {upperflag}",
            )


def check_direction_bins(args: argparse.Namespace) -> None:
    """Stop when --plotviewingangle names a bin that does not exist, before the command reads the spectra."""
    dirbins = getattr(args, "plotviewingangle", None)
    if not dirbins:
        return
    ndirbins = get_viewingdirectionbincount()
    specialbins = {-1, -2} if getattr(args, "allowalldirbins", False) else {-1}
    if badbins := [dirbin for dirbin in dirbins if dirbin not in specialbins and not 0 <= dirbin < ndirbins]:
        specialtext = (
            "-1 for all the directions, -2 for each bin," if -2 in specialbins else "-1 for all the directions,"
        )
        exit_with_error(
            f"--plotviewingangle {' '.join(map(str, badbins))} names no direction bin",
            f"Give {specialtext} or a bin from 0 to {ndirbins - 1}",
        )


def resolve_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argsraw: "Sequence[str] | None" = None,
    kwargs: "Collection[str] | None" = None,
) -> None:
    """Check the arguments of a command and complete them after the parse, before the command reads data.

    The dispatcher and parse_cli_args both call this function, thus each entry point makes the same checks. A check
    that needs only the arguments belongs here, because then a mistake costs no time.
    """
    assert isinstance(parser, SuggestingArgumentParser), "every parser of a command is this class"
    check_time_selection(parser, args, argsraw, kwargs)
    check_range_order(args)
    check_direction_bins(args)
    resolve_output_argument(args)
    resolve_yscale(args)


# the dests that hold the model paths of a command. plotspectra names its dest specpath, because a path can also name
# a reference spectrum
MODELPATH_DESTS: t.Final = ("modelpath", "specpath")


def addarg_common(parser: argparse.ArgumentParser) -> None:
    """Add the arguments that every command takes: --quiet, and --explain for a command that reads a model.

    The dispatcher and parse_cli_args both call this function, thus no module declares these arguments.
    """
    addarg_quiet(parser)
    if any(action.dest in MODELPATH_DESTS for action in parser._actions):  # ruff:ignore[private-member-access]
        arggroup(parser, "Output").add_argument(
            "--explain",
            action="store_true",
            help=(
                "Print the model folders, the selection, and the output files that the arguments give, then stop."
                " The command then reads no data"
            ),
        )


def print_explanation(
    parser: argparse.ArgumentParser, args: argparse.Namespace, parsedargs: argparse.Namespace
) -> None:
    """Print what the command reads and writes, as the arguments give it. Read no data.

    parsedargs holds the arguments before resolve_args, thus a value that resolve_args sets, e.g. the scale "auto" of
    the y axis, is not a value of the user. The lines give each argument of the user with its value after the
    conversion of the units. They also give:

    - the model folders;
    - the timesteps of the time selection;
    - the direction bins;
    - the output file.
    """
    print_heading("The arguments select these data and files. --explain stops the command before it reads the data")
    for action in parser._actions:  # ruff:ignore[private-member-access]
        if action.dest in {"help", "explain", "quiet", *MODELPATH_DESTS} or not hasattr(args, action.dest):
            continue
        parsedvalue = getattr(parsedargs, action.dest, None)
        if parsedvalue is not None and parsedvalue != parser.get_default(action.dest):
            flag = get_shown_flags(action)[0] if action.option_strings else action.dest
            print_detail(f"{flag} = {getattr(args, action.dest)}")

    # normalize_path_list gives the working folder for None, thus a command with no such dest gives no path
    for dest in (dest for dest in MODELPATH_DESTS if hasattr(args, dest)):
        for modelpath in normalize_path_list(getattr(args, dest)):
            explain_model_selection(modelpath, args)

    if (outputfile := getattr(args, "outputfile", None)) is not None:
        existstext = (
            " (the file exists, thus the command asks before it overwrites it)" if Path(outputfile).exists() else ""
        )
        print_detail(f"Output: {outputfile}{existstext}")


def explain_model_selection(modelpath: Path, args: argparse.Namespace) -> None:
    """Print the model folder, the timesteps of the time selection, and the direction bins for one model path."""
    if is_remote_path(modelpath):
        print_detail(f"Model: {modelpath} (a remote model. --explain reads no file of it)")
        return
    modelfolder = get_model_folder(modelpath)
    if not modelfolder.is_dir():
        print_detail(f"Model: {modelfolder} does not exist")
        return
    if not folder_is_artis_model(modelfolder):
        # plotspectra also reads a reference spectrum, which is a file outside a model folder
        print_detail(f"Path: {modelpath} (no model folder, because {modelfolder} holds no input.txt)")
        return
    print_detail(f"Model: {modelfolder}")

    timeselection = {name: getattr(args, name, None) for name in ("timestep", "timemin", "timemax", "timedays")}
    if any(value is not None for value in timeselection.values()):
        try:
            timestepmin, timestepmax, timemin, timemax = get_time_range(
                modelfolder,
                timestep_range_str=timeselection["timestep"],
                timemin=timeselection["timemin"],
                timemax=timeselection["timemax"],
                timedays_range_str=timeselection["timedays"],
            )
        except (OSError, ValueError) as exc:
            print_detail(f"Time: the selection gives an error: {exc}")
        else:
            print_detail(
                f"Time: timesteps {timestepmin} to {timestepmax}, which cover {timemin:.3f} to {timemax:.3f} d"
            )

    if dirbins := [dirbin for dirbin in getattr(args, "plotviewingangle", None) or [] if dirbin != -2]:
        try:
            labels = get_dirbin_labels(
                dirbins,
                average_over_phi=getattr(args, "average_over_phi_angle", False),
                average_over_theta=getattr(args, "average_over_theta_angle", False),
            )
        except ValueError as exc:
            print_detail(f"Direction: the selection gives an error: {exc}")
        else:
            for dirbin, label in labels.items():
                print_detail(f"Direction bin {dirbin}: {label}")


def parse_cli_args(
    addargsfunc: Callable[[argparse.ArgumentParser], None],
    description: str | None,
    args: argparse.Namespace | None,
    argsraw: Sequence[str] | None = None,
    kwargs: dict[str, t.Any] | None = None,
) -> argparse.Namespace:
    """Return args if the caller parsed them, or else parse the command line with the options of addargsfunc.

    The keyword arguments replace the parser defaults, and argsraw then gives the other arguments as text. A call
    that gives keyword arguments and no argsraw reads no command line, because sys.argv holds a different command.
    """
    if args is not None:
        return args

    parser = SuggestingArgumentParser(formatter_class=CustomArgHelpFormatter, description=description)
    addargsfunc(parser)
    # the dispatcher adds --quiet and --explain to the parser that it builds, thus a direct call needs them here
    addarg_common(parser)
    parser.add_standard_flags()
    kwargs = kwargs or {}
    set_args_from_dict(parser, kwargs)
    tokens = [] if argsraw is None and kwargs else argsraw
    args = parser.parse_args(separate_trailing_folders(tokens))
    parsedargs = copy.copy(args)
    resolve_args(parser, args, tokens, kwargs)
    if getattr(args, "explain", False):
        print_explanation(parser, args, parsedargs)
        raise SystemExit(0)

    return args


def takes_a_list(action: argparse.Action) -> bool:
    """Return whether the command line gives this argument a list of values."""
    return action.nargs in {"*", "+"} or (isinstance(action.nargs, int) and action.nargs > 1)


def convert_keyword_items(arg: argparse.Action, items: "Sequence[t.Any]") -> list[t.Any]:
    """Return the items of a list keyword after the type of the argument, as the command line converts them.

    argparse converts a default of one text alone, thus label=["default", "B"] kept the text "default". An item that
    is not a text comes from Python code, thus it stays as it is.
    """
    converter = arg.type
    flag = get_shown_flags(arg)[0] if arg.option_strings else arg.dest
    try:
        converted = [converter(item) if callable(converter) and isinstance(item, str) else item for item in items]
    except (argparse.ArgumentTypeError, ValueError) as exc:
        msg = f"{flag}: {exc}"
        raise ValueError(msg) from exc

    return arg.with_default_colours(converted) if isinstance(arg, KeepDefaultColors) else converted


def set_args_from_dict(parser: argparse.ArgumentParser, kwargs: dict[str, t.Any]) -> None:
    """Set argparse defaults from a dictionary.

    A name that this command does not take raises. addarg_unsupported declares an old name, so that a
    user of the command line gets a message. Such a name is no argument of this command, thus a keyword
    of that name also raises.

    A keyword that names a destination takes priority over an option alias.
    """
    kwargs = kwargs.copy()  # keys are renamed to argument dests below, so don't mutate the caller's dict
    realactions = [
        action
        for action in parser._actions  # ruff:ignore[private-member-access]
        if not isinstance(action, UnsupportedArgument)
    ]
    destinations = {arg.dest for arg in realactions}
    # set_defaults expects the dest of an argument. A keyword can also name an option string of the argument
    for arg in realactions:
        aliases = [flag.lstrip("-") for flag in arg.option_strings if flag.lstrip("-") not in destinations]
        names = list(dict.fromkeys(name for name in (arg.dest, *aliases) if name in kwargs))
        if len(names) > 1:
            msg = f"The keywords {', '.join(names)} name one argument, thus give only one of them"
            raise ValueError(msg)
        if names and names[0] != arg.dest:
            kwargs[arg.dest] = kwargs.pop(names[0])

    # an option that reads a list gets a list from the command line, thus main(plotviewingangle=0)
    # must mean the same as -plotviewingangle 0
    for arg in realactions:
        value = kwargs.get(arg.dest)
        # -dashes reads one tuple of numbers for each series, thus a single tuple of numbers is one item and not a
        # list. The type of -dashes is a wrapper of dashes_arg, thus the test reads the dest
        istupleitem = (
            arg.dest == "dashes"
            and isinstance(value, tuple)
            and all(isinstance(length, int | float) for length in value)
        )
        if value is not None and isinstance(arg, CellListAction):
            kwargs[arg.dest] = get_cell_list(value)
        elif value is not None and takes_a_list(arg):
            istextlist = isinstance(value, list | tuple) and not istupleitem  # pyrefly: ignore[implicit-any-type-argument]
            kwargs[arg.dest] = convert_keyword_items(arg, value if istextlist else [value])

    # set_defaults gives each argument of this dest the new default, thus the colours of -color are read first
    parser.set_defaults(**kwargs)
    # every argument takes required=False. A keyword argument can give the value instead, thus a
    # required argument would give an error for a value that the caller did supply
    for arg in realactions:
        if arg.default is not None:
            arg.required = False

    if unknown := {k: v for k, v in kwargs.items() if k not in (arg.dest for arg in realactions)}:
        msg = f"Unknown argument names: {unknown}"
        raise ValueError(msg)


def normalize_path_list(paths: PathArg) -> list[Path]:
    """Return a flat list of Paths from one path or a nested sequence of paths, or the working folder for no path.

    A remote path gets the canonical form of get_canonical_path, thus its parent keeps the host.
    """

    def to_path(path: str | Path) -> Path:
        return get_canonical_path(model_path_from_text(path) if isinstance(path, str) else Path(path))

    if not paths:
        return [Path()]
    if isinstance(paths, str | Path):
        return [to_path(paths)]
    return [to_path(p) for p in flatten_list(list(paths))]
