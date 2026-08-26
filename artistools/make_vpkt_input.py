"""Create or interactively edit a vpkt.txt virtual packet configuration file for an ARTIS simulation."""

import argparse
import dataclasses as dc
import sys
import typing as t
from collections.abc import Callable
from collections.abc import Sequence

import artistools as at
from artistools.misc import addarg_outputfile
from artistools.misc import print_warning
from artistools.misc import resolve_outputfile

defaultoutputfile = "vpkt.txt"

OPACITY_CHOICE_HELP = (
    "0 full opacity, -1 no lines, -2 no bound-free, -3 no free-free, -4 no electron scattering,"
    " or a positive atomic number to exclude that element's bound-bound opacity"
)

# defaults of the vpkt.h constants that read_vpktparameterfile() asserts against, so only warn
ARTIS_VSPEC_TIMEMIN_DAYS = 3.0
ARTIS_VSPEC_TIMEMAX_DAYS = 8.0
ARTIS_VSPEC_LAMBDAMIN_ANGSTROMS = 3500.0
ARTIS_VSPEC_LAMBDAMAX_ANGSTROMS = 10000.0


def fmtnum(value: float) -> str:
    """Format a number for vpkt.txt, dropping the decimal part when it is a whole number."""
    return str(int(value)) if value.is_integer() else str(value)


@dc.dataclass
class VpktConfig:
    """The settings held in a vpkt.txt file, in the order they appear."""

    directions_costheta_phi: list[tuple[float, float]] = dc.field(
        default_factory=lambda: [(1.0, 0.0), (0.0, 0.0), (-1.0, 0.0)]
    )
    opacityexclusions: list[int] = dc.field(default_factory=list)
    override_tminmax: bool = False
    vspec_tmin_in_days: float = 0.2
    vspec_tmax_in_days: float = 1.5
    custom_lambda_ranges: list[tuple[float, float]] = dc.field(default_factory=list)
    override_thickcell_tau: bool = True
    cell_is_optically_thick_vpkt: float = 100.0
    tau_max_vpkt: float = 10.0
    vgrid_on: bool = False
    tmin_vgrid_in_days: float = 0.2
    tmax_vgrid_in_days: float = 1.5
    vgrid_lambda_ranges: list[tuple[float, float]] = dc.field(default_factory=lambda: [(3500.0, 6000.0)])


def parse_pair(strpair: str, label: str, example: str) -> tuple[float, float]:
    """Parse a comma-separated pair of numbers, e.g. '-1,0'."""
    msg = f"{label} {strpair!r} must be two numbers separated by a comma, e.g. {example!r}"
    parts = strpair.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(msg)
    try:
        first, second = (float(x) for x in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(msg) from exc

    return first, second


def parse_direction(strdirection: str) -> tuple[float, float]:
    """Parse a 'costheta,phi' viewing direction, e.g. '-1,0'."""
    costheta, phi = parse_pair(strdirection, "Viewing direction", "-1,0")
    if not -1.0 <= costheta <= 1.0:
        msg = f"costheta {costheta} in viewing direction {strdirection!r} must be between -1 and 1"
        raise argparse.ArgumentTypeError(msg)

    return costheta, phi


def parse_directions(strdirections: str) -> list[tuple[float, float]]:
    """Parse whitespace-separated 'costheta,phi' viewing directions, e.g. '1,0 0,0 -1,0'.

    One string rather than a list, because argparse reads a bare '-1,0' as an option name and would
    put the negative costheta half of the sky out of reach.
    """
    directions = [parse_direction(token) for token in strdirections.split()]
    if not directions:
        msg = "At least one viewing direction is required, e.g. '1,0'"
        raise argparse.ArgumentTypeError(msg)

    return directions


def parse_lambda_range(strrange: str) -> tuple[float, float]:
    """Parse a 'lambdamin,lambdamax' wavelength range in Angstroms."""
    lambdamin, lambdamax = parse_pair(strrange, "Wavelength range", "3500,6000")
    if lambdamin >= lambdamax:
        msg = f"Wavelength range {strrange!r} must have lambdamin < lambdamax"
        raise argparse.ArgumentTypeError(msg)

    return lambdamin, lambdamax


def parse_lambda_ranges(strranges: str) -> list[tuple[float, float]]:
    """Parse whitespace-separated 'lambdamin,lambdamax' ranges, or an empty string for none."""
    return [parse_lambda_range(token) for token in strranges.split()]


def first_opacity_choice_error(firstchoice: int) -> str:
    """Return the message for an opacity list that ARTIS would abort on."""
    return (
        f"The first opacity choice must be 0 (full opacity), not {firstchoice}."
        " ARTIS asserts opacityexclusions[0] == 0 and will abort otherwise."
    )


def parse_opacityexclusions(strexclusions: str) -> list[int]:
    """Parse whitespace-separated opacity choices, or an empty string for none."""
    try:
        exclusions = [int(token) for token in strexclusions.split()]
    except ValueError as exc:
        msg = f"Opacity choices {strexclusions!r} must be whitespace-separated integers ({OPACITY_CHOICE_HELP})"
        raise argparse.ArgumentTypeError(msg) from exc

    if exclusions and exclusions[0] != 0:
        raise argparse.ArgumentTypeError(first_opacity_choice_error(exclusions[0]))

    return exclusions


def parse_bool(strbool: str) -> bool:
    """Parse a yes/no answer."""
    if (reply := strbool.strip().lower()) in {"y", "yes", "true", "1", "on"}:
        return True
    if reply in {"n", "no", "false", "0", "off"}:
        return False

    msg = f"Answer {strbool!r} must be yes or no"
    raise argparse.ArgumentTypeError(msg)


def fatal_config_error(config: VpktConfig) -> str | None:
    """Return the reason ARTIS would abort on this config, or None if it would accept it."""
    if config.opacityexclusions and config.opacityexclusions[0] != 0:
        return first_opacity_choice_error(config.opacityexclusions[0])

    for i, (costheta, _) in enumerate(config.directions_costheta_phi):
        if abs(costheta) > 1:
            return f"Observer direction {i} has costheta {fmtnum(costheta)}, which is outside [-1, 1]"

    for lambdamin, lambdamax in config.custom_lambda_ranges + config.vgrid_lambda_ranges:
        if lambdamin >= lambdamax:
            return f"Wavelength range [{fmtnum(lambdamin)}, {fmtnum(lambdamax)}] must have lambdamin < lambdamax"

    return None


def check_config(config: VpktConfig) -> list[str]:
    """Return warnings for settings outside the limits vpkt.h was compiled with.

    These only warn because a build may legitimately have been made with different constants.
    """
    warnings = []
    if config.override_tminmax and not (
        ARTIS_VSPEC_TIMEMIN_DAYS <= config.vspec_tmin_in_days <= config.vspec_tmax_in_days <= ARTIS_VSPEC_TIMEMAX_DAYS
    ):
        warnings.append(
            f"time window [{fmtnum(config.vspec_tmin_in_days)}, {fmtnum(config.vspec_tmax_in_days)}] d falls outside"
            f" the default VSPEC_TIMEMIN/VSPEC_TIMEMAX of [{fmtnum(ARTIS_VSPEC_TIMEMIN_DAYS)},"
            f" {fmtnum(ARTIS_VSPEC_TIMEMAX_DAYS)}] d in vpkt.h"
        )

    for lambdamin, lambdamax in config.custom_lambda_ranges:
        if not ARTIS_VSPEC_LAMBDAMIN_ANGSTROMS <= lambdamin < lambdamax <= ARTIS_VSPEC_LAMBDAMAX_ANGSTROMS:
            warnings.append(
                f"wavelength range [{fmtnum(lambdamin)}, {fmtnum(lambdamax)}] A falls outside the default"
                f" [{fmtnum(ARTIS_VSPEC_LAMBDAMIN_ANGSTROMS)}, {fmtnum(ARTIS_VSPEC_LAMBDAMAX_ANGSTROMS)}] A"
                " set by VSPEC_NUMIN/VSPEC_NUMAX in vpkt.h"
            )

    return warnings


def format_vpkt_input(config: VpktConfig) -> str:
    """Return the contents of a vpkt.txt file."""
    if fatalerror := fatal_config_error(config):
        raise ValueError(fatalerror)

    str_opacityexclusions = (
        f"{len(config.opacityexclusions)} " + " ".join(str(x) for x in config.opacityexclusions)
        if config.opacityexclusions
        else ""
    )
    str_custom_lambda_ranges = (
        (
            f" {len(config.custom_lambda_ranges)} "
            + " ".join(f"{fmtnum(lmin)} {fmtnum(lmax)}" for lmin, lmax in config.custom_lambda_ranges)
        )
        if config.custom_lambda_ranges
        else ""
    )

    return (
        f"{len(config.directions_costheta_phi)}\n"
        f"{' '.join(fmtnum(costheta) for costheta, _ in config.directions_costheta_phi)}\n"
        f"{' '.join(fmtnum(phi) for _, phi in config.directions_costheta_phi)}\n"
        f"{bool(config.opacityexclusions):d} {str_opacityexclusions}\n"
        f"{config.override_tminmax:d} {fmtnum(config.vspec_tmin_in_days)} {fmtnum(config.vspec_tmax_in_days)}\n"
        f"{bool(config.custom_lambda_ranges):d}{str_custom_lambda_ranges}\n"
        f"{config.override_thickcell_tau:d} {fmtnum(config.cell_is_optically_thick_vpkt)}\n"
        f"{fmtnum(config.tau_max_vpkt)}\n"
        f"{config.vgrid_on:d}\n"
        f"{fmtnum(config.tmin_vgrid_in_days)} {fmtnum(config.tmax_vgrid_in_days)}\n"
        f"{len(config.vgrid_lambda_ranges)} "
        + " ".join(f"{fmtnum(lmin)} {fmtnum(lmax)}" for lmin, lmax in config.vgrid_lambda_ranges)
    )


class VpktTokenReader:
    """Hand out whitespace-separated tokens, so that any file ARTIS reads with fscanf is accepted."""

    def __init__(self, contents: str) -> None:
        """Split the file contents into tokens, ready to be consumed in order."""
        self.tokens = contents.split()
        self.pos = 0

    def next_value[T](self, converter: Callable[[str], T], typename: str, description: str) -> T:
        """Return the next token converted by converter, naming what was expected on failure."""
        if self.pos >= len(self.tokens):
            msg = f"vpkt.txt ended while reading {description}"
            raise ValueError(msg)

        token = self.tokens[self.pos]
        self.pos += 1
        try:
            return converter(token)
        except ValueError as exc:
            msg = f"vpkt.txt has {token!r} where {description} should be a {typename}"
            raise ValueError(msg) from exc

    def next_int(self, description: str) -> int:
        """Return the next token as an integer."""
        return self.next_value(int, "integer", description)

    def next_float(self, description: str) -> float:
        """Return the next token as a float."""
        return self.next_value(float, "number", description)

    def next_lambda_ranges(self, count: int, description: str) -> list[tuple[float, float]]:
        """Return count consecutive lambdamin/lambdamax pairs."""
        return [
            (self.next_float(f"{description} lambdamin"), self.next_float(f"{description} lambdamax"))
            for _ in range(count)
        ]

    @property
    def exhausted(self) -> bool:
        """Whether every token has been consumed."""
        return self.pos >= len(self.tokens)


def parse_vpkt_input(contents: str) -> VpktConfig:
    """Parse the contents of a vpkt.txt file, following the same order as the ARTIS reader."""
    reader = VpktTokenReader(contents)

    ndirections = reader.next_int("the number of observer directions")
    if ndirections < 1:
        msg = f"vpkt.txt must have at least one observer direction, but declares {ndirections}"
        raise ValueError(msg)
    costhetas = [reader.next_float(f"costheta of direction {i}") for i in range(ndirections)]
    phis = [reader.next_float(f"phi of direction {i}") for i in range(ndirections)]

    # any flag other than 1 means one full-opacity spectrum, which an empty list stands for here
    opacityexclusions: list[int] = []
    if reader.next_int("the custom opacity list flag") == 1:
        nspectraperobsdir = reader.next_int("the number of opacity choices")
        opacityexclusions = [reader.next_int(f"opacity choice {i}") for i in range(nspectraperobsdir)]

    override_tminmax = reader.next_int("the time window override flag")
    vspec_tmin_in_days = reader.next_float("the time window start")
    vspec_tmax_in_days = reader.next_float("the time window end")

    custom_lambda_ranges: list[tuple[float, float]] = []
    if reader.next_int("the custom wavelength range flag") == 1:
        nwavelengthranges = reader.next_int("the number of wavelength ranges")
        custom_lambda_ranges = reader.next_lambda_ranges(nwavelengthranges, "wavelength range")

    override_thickcell_tau = reader.next_int("the thick cell override flag")
    cell_is_optically_thick_vpkt = reader.next_float("the thick cell optical depth")
    tau_max_vpkt = reader.next_float("tau_max_vpkt")
    vgrid_on = bool(reader.next_int("the velocity grid map flag"))

    config = VpktConfig(
        directions_costheta_phi=list(zip(costhetas, phis, strict=True)),
        opacityexclusions=opacityexclusions,
        override_tminmax=bool(override_tminmax),
        vspec_tmin_in_days=vspec_tmin_in_days,
        vspec_tmax_in_days=vspec_tmax_in_days,
        custom_lambda_ranges=custom_lambda_ranges,
        override_thickcell_tau=bool(override_thickcell_tau),
        cell_is_optically_thick_vpkt=cell_is_optically_thick_vpkt,
        tau_max_vpkt=tau_max_vpkt,
        vgrid_on=vgrid_on,
    )

    # ARTIS stops reading here when the map is off, so the block may legitimately be absent
    if vgrid_on or not reader.exhausted:
        config.tmin_vgrid_in_days = reader.next_float("the velocity grid map start time")
        config.tmax_vgrid_in_days = reader.next_float("the velocity grid map end time")
        ngridranges = reader.next_int("the number of velocity grid wavelength ranges")
        config.vgrid_lambda_ranges = reader.next_lambda_ranges(ngridranges, "velocity grid wavelength range")

    return config


def show_pairs(pairs: Sequence[tuple[float, float]]) -> str:
    """Render number pairs in the comma-separated form the prompts and arguments accept."""
    return " ".join(f"{fmtnum(first)},{fmtnum(second)}" for first, second in pairs)


def show_bool(value: bool) -> str:
    """Render a boolean as the yes/no that parse_bool reads back."""
    return "yes" if value else "no"


@dc.dataclass(frozen=True, slots=True)
class VpktField:
    """One editable setting: which attribute it sets, how to show it, and how to read it back."""

    attr: str
    prompt: str
    parse: Callable[[str], t.Any]
    show: Callable[[t.Any], str]


def get_editable_fields() -> list[VpktField]:
    """Return the settings offered by the interactive editor, in file order."""
    return [
        VpktField("directions_costheta_phi", "Viewing directions as costheta,phi pairs", parse_directions, show_pairs),
        VpktField(
            "opacityexclusions",
            f"Opacity choices ({OPACITY_CHOICE_HELP})",
            parse_opacityexclusions,
            lambda values: " ".join(str(x) for x in values),
        ),
        VpktField("override_tminmax", "Restrict virtual packets to a time window?", parse_bool, show_bool),
        VpktField("vspec_tmin_in_days", "Virtual packet time window start [days]", float, fmtnum),
        VpktField("vspec_tmax_in_days", "Virtual packet time window end [days]", float, fmtnum),
        VpktField(
            "custom_lambda_ranges",
            "Custom wavelength ranges as lambdamin,lambdamax pairs [Angstroms]",
            parse_lambda_ranges,
            show_pairs,
        ),
        VpktField("override_thickcell_tau", "Skip virtual packets in optically thick cells?", parse_bool, show_bool),
        VpktField("cell_is_optically_thick_vpkt", "Cell optical depth counted as thick", float, fmtnum),
        VpktField("tau_max_vpkt", "Maximum optical depth before a virtual packet is discarded", float, fmtnum),
        VpktField("vgrid_on", "Produce a velocity grid map?", parse_bool, show_bool),
        VpktField("tmin_vgrid_in_days", "Velocity grid map start [days]", float, fmtnum),
        VpktField("tmax_vgrid_in_days", "Velocity grid map end [days]", float, fmtnum),
        VpktField(
            "vgrid_lambda_ranges",
            "Velocity grid map wavelength ranges as lambdamin,lambdamax pairs [Angstroms]",
            parse_lambda_ranges,
            show_pairs,
        ),
    ]


def edit_config_interactively(config: VpktConfig, promptfunc: Callable[[str], str] = input) -> VpktConfig:
    """Ask for each setting in turn, keeping the current value when the reply is empty.

    An invalid reply is reported and the question repeated, so the caller sees no parse failures.
    """
    print("Press enter to keep the current value shown in brackets. Enter a single '-' to clear a list.")

    for field in get_editable_fields():
        current = getattr(config, field.attr)
        while True:
            reply = promptfunc(f"{field.prompt} [{field.show(current)}]: ").strip()
            if not reply:
                break
            try:
                setattr(config, field.attr, field.parse("" if reply == "-" else reply))
            except (argparse.ArgumentTypeError, ValueError) as exc:
                print(f"  {exc}")
            else:
                break

    return config


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    parser.add_argument(
        "-directions",
        type=parse_directions,
        default=None,
        help=(
            "Viewing directions as whitespace-separated costheta,phi pairs in one quoted string."
            " Use an equals sign when the first value is negative, e.g. -directions='-1,0 0,0 1,0'"
        ),
    )
    parser.add_argument(
        "-opacityexclusions", type=parse_opacityexclusions, default=None, help=f"Opacity choices: {OPACITY_CHOICE_HELP}"
    )
    parser.add_argument(
        "--override-tminmax",
        action="store_true",
        default=None,
        help="Restrict virtual packets to the time window given by -vspec-tmin and -vspec-tmax",
    )
    parser.add_argument("-vspec-tmin", type=float, default=None, help="Start of the virtual packet time window [days]")
    parser.add_argument("-vspec-tmax", type=float, default=None, help="End of the virtual packet time window [days]")
    parser.add_argument(
        "-lambdaranges",
        type=parse_lambda_ranges,
        default=None,
        help="Custom wavelength ranges as whitespace-separated lambdamin,lambdamax pairs in Angstroms",
    )
    parser.add_argument(
        "--no-override-thickcell-tau",
        dest="override_thickcell_tau",
        action="store_false",
        default=None,
        help="Create virtual packets even in cells more optically thick than -cell-is-optically-thick",
    )
    parser.add_argument(
        "-cell-is-optically-thick",
        type=float,
        default=None,
        help="Cell optical depth above which virtual packets are not created",
    )
    parser.add_argument(
        "-tau-max", type=float, default=None, help="Maximum optical depth before a virtual packet is discarded"
    )
    parser.add_argument("--vgrid", action="store_true", default=None, help="Produce a velocity grid map")
    parser.add_argument("-vgrid-tmin", type=float, default=None, help="Start of the velocity grid map range [days]")
    parser.add_argument("-vgrid-tmax", type=float, default=None, help="End of the velocity grid map range [days]")
    parser.add_argument(
        "-vgrid-lambdaranges",
        type=parse_lambda_ranges,
        default=None,
        help="Velocity grid map wavelength ranges as whitespace-separated lambdamin,lambdamax pairs in Angstroms",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Do not prompt for settings. Prompting is also skipped when stdin is not a terminal.",
    )
    addarg_outputfile(parser, helptext=f"Path/filename for the output file (default {defaultoutputfile})")


def apply_args_to_config(config: VpktConfig, args: argparse.Namespace) -> VpktConfig:
    """Apply the settings that were given explicitly on the command line, leaving the rest alone."""
    argname_of_attr = {
        "directions_costheta_phi": "directions",
        "opacityexclusions": "opacityexclusions",
        "override_tminmax": "override_tminmax",
        "vspec_tmin_in_days": "vspec_tmin",
        "vspec_tmax_in_days": "vspec_tmax",
        "custom_lambda_ranges": "lambdaranges",
        "override_thickcell_tau": "override_thickcell_tau",
        "cell_is_optically_thick_vpkt": "cell_is_optically_thick",
        "tau_max_vpkt": "tau_max",
        "vgrid_on": "vgrid",
        "tmin_vgrid_in_days": "vgrid_tmin",
        "tmax_vgrid_in_days": "vgrid_tmax",
        "vgrid_lambda_ranges": "vgrid_lambdaranges",
    }
    for attr, argname in argname_of_attr.items():
        value = getattr(args, argname)
        if value is not None:
            setattr(config, attr, value)

    return config


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Create or interactively edit a vpkt.txt virtual packet configuration file for an ARTIS simulation."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    outputfile = resolve_outputfile(args.outputfile, defaultoutputfile)

    if outputfile.is_file():
        config = parse_vpkt_input(outputfile.read_text(encoding="utf-8"))
        print(f"Read existing {outputfile}")
        # report rather than raise, so a file that ARTIS would reject can still be loaded and repaired
        if fatalerror := fatal_config_error(config):
            print(f"PROBLEM in {outputfile}: {fatalerror}")
    else:
        config = VpktConfig()
        print(f"{outputfile} does not exist, so starting from the default settings")

    config = apply_args_to_config(config, args)

    if not args.non_interactive:
        if sys.stdin.isatty():
            config = edit_config_interactively(config)
        else:
            print("stdin is not a terminal, so keeping the settings above without prompting")

    for warning in check_config(config):
        print_warning(f"{warning}. ARTIS will abort unless it was built with matching constants.")

    outputfile.write_text(format_vpkt_input(config), encoding="utf-8")
    at.print_saved(outputfile)


if __name__ == "__main__":
    main()
