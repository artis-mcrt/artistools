"""Create or interactively edit a vpkt.txt virtual packet configuration file for an ARTIS simulation."""

import argparse
import dataclasses as dc
import sys
import typing as t
from collections.abc import Callable
from collections.abc import Sequence

import artistools as at
from artistools.misc import add_outputfile_arg
from artistools.misc import resolve_outputfile

defaultoutputfile = "vpkt.txt"

OPACITY_CHOICE_HELP = (
    "0 full opacity, -1 no lines, -2 no bound-free, -3 no free-free, -4 no electron scattering,"
    " or a positive atomic number to exclude that element's bound-bound opacity"
)


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
    nrange_grid: int = 1
    vgrid_lambda_min: float = 3500.0
    vgrid_lambda_max: float = 6000.0


def parse_direction(strdirection: str) -> tuple[float, float]:
    """Parse a 'costheta,phi' viewing direction, e.g. '-1,0'."""
    parts = strdirection.split(",")
    if len(parts) != 2:
        msg = f"Viewing direction {strdirection!r} must be given as 'costheta,phi', e.g. '-1,0'"
        raise argparse.ArgumentTypeError(msg)
    try:
        costheta, phi = (float(x) for x in parts)
    except ValueError as exc:
        msg = f"Viewing direction {strdirection!r} must be two numbers separated by a comma, e.g. '-1,0'"
        raise argparse.ArgumentTypeError(msg) from exc

    if not -1.0 <= costheta <= 1.0:
        msg = f"costheta {costheta} in viewing direction {strdirection!r} must be between -1 and 1"
        raise argparse.ArgumentTypeError(msg)

    return costheta, phi


def parse_directions(strdirections: str) -> list[tuple[float, float]]:
    """Parse whitespace-separated 'costheta,phi' viewing directions, e.g. '1,0 0,0 -1,0'.

    This takes one string rather than a list of them because argparse reads a bare '-1,0' as an
    option name, so a list would make the negative costheta half of the sky unreachable.
    """
    directions = [parse_direction(token) for token in strdirections.split()]
    if not directions:
        msg = "At least one viewing direction is required, e.g. '1,0'"
        raise argparse.ArgumentTypeError(msg)

    return directions


def parse_lambda_range(strrange: str) -> tuple[float, float]:
    """Parse a 'lambdamin,lambdamax' wavelength range in Angstroms."""
    parts = strrange.split(",")
    if len(parts) != 2:
        msg = f"Wavelength range {strrange!r} must be given as 'lambdamin,lambdamax', e.g. '3500,6000'"
        raise argparse.ArgumentTypeError(msg)
    try:
        lambdamin, lambdamax = (float(x) for x in parts)
    except ValueError as exc:
        msg = f"Wavelength range {strrange!r} must be two numbers separated by a comma, e.g. '3500,6000'"
        raise argparse.ArgumentTypeError(msg) from exc

    if lambdamin >= lambdamax:
        msg = f"Wavelength range {strrange!r} must have lambdamin < lambdamax"
        raise argparse.ArgumentTypeError(msg)

    return lambdamin, lambdamax


def parse_lambda_ranges(strranges: str) -> list[tuple[float, float]]:
    """Parse whitespace-separated 'lambdamin,lambdamax' ranges, or an empty string for none."""
    return [parse_lambda_range(token) for token in strranges.split()]


def parse_opacityexclusions(strexclusions: str) -> list[int]:
    """Parse whitespace-separated opacity choices, or an empty string for none."""
    try:
        return [int(token) for token in strexclusions.split()]
    except ValueError as exc:
        msg = f"Opacity choices {strexclusions!r} must be whitespace-separated integers ({OPACITY_CHOICE_HELP})"
        raise argparse.ArgumentTypeError(msg) from exc


def parse_bool(strbool: str) -> bool:
    """Parse a yes/no answer."""
    if (reply := strbool.strip().lower()) in {"y", "yes", "true", "1", "on"}:
        return True
    if reply in {"n", "no", "false", "0", "off"}:
        return False

    msg = f"Answer {strbool!r} must be yes or no"
    raise argparse.ArgumentTypeError(msg)


def format_vpkt_input(config: VpktConfig | None = None) -> str:
    """Return the contents of a vpkt.txt file."""
    if config is None:
        config = VpktConfig()

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
        # only one wavelength range is written here, although the file format allows several
        f"{config.nrange_grid} {fmtnum(config.vgrid_lambda_min)} {fmtnum(config.vgrid_lambda_max)}"
    )


def parse_vpkt_input(contents: str) -> VpktConfig:
    """Parse the contents of a vpkt.txt file."""
    lines = contents.splitlines()
    if len(lines) < 11:
        msg = f"vpkt.txt must have at least 11 lines, but this one has {len(lines)}"
        raise ValueError(msg)

    ndirections = int(lines[0].split()[0])
    costhetas = [float(x) for x in lines[1].split()]
    phis = [float(x) for x in lines[2].split()]
    if len(costhetas) != ndirections or len(phis) != ndirections:
        msg = (
            f"vpkt.txt declares {ndirections} viewing directions but lists"
            f" {len(costhetas)} costheta and {len(phis)} phi values"
        )
        raise ValueError(msg)

    opacityexclusions: list[int] = []
    if int(lines[3].split()[0]):
        exclusiontokens = [int(x) for x in lines[3].split()[1:]]
        nexclusions, opacityexclusions = exclusiontokens[0], exclusiontokens[1:]
        if len(opacityexclusions) != nexclusions:
            msg = f"vpkt.txt declares {nexclusions} opacity choices but lists {len(opacityexclusions)}"
            raise ValueError(msg)

    override_tminmax, vspec_tmin, vspec_tmax = lines[4].split()

    custom_lambda_ranges: list[tuple[float, float]] = []
    if int(lines[5].split()[0]):
        rangetokens = lines[5].split()[1:]
        nranges = int(rangetokens[0])
        bounds = [float(x) for x in rangetokens[1:]]
        if len(bounds) != 2 * nranges:
            msg = f"vpkt.txt declares {nranges} wavelength ranges but lists {len(bounds)} bounds"
            raise ValueError(msg)
        custom_lambda_ranges = list(zip(bounds[::2], bounds[1::2], strict=True))

    override_thickcell_tau, cell_is_optically_thick = lines[6].split()
    tmin_vgrid, tmax_vgrid = lines[9].split()
    nrange_grid, vgrid_lambda_min, vgrid_lambda_max = lines[10].split()

    return VpktConfig(
        directions_costheta_phi=list(zip(costhetas, phis, strict=True)),
        opacityexclusions=opacityexclusions,
        override_tminmax=bool(int(override_tminmax)),
        vspec_tmin_in_days=float(vspec_tmin),
        vspec_tmax_in_days=float(vspec_tmax),
        custom_lambda_ranges=custom_lambda_ranges,
        override_thickcell_tau=bool(int(override_thickcell_tau)),
        cell_is_optically_thick_vpkt=float(cell_is_optically_thick),
        tau_max_vpkt=float(lines[7].split()[0]),
        vgrid_on=bool(int(lines[8].split()[0])),
        tmin_vgrid_in_days=float(tmin_vgrid),
        tmax_vgrid_in_days=float(tmax_vgrid),
        nrange_grid=int(nrange_grid),
        vgrid_lambda_min=float(vgrid_lambda_min),
        vgrid_lambda_max=float(vgrid_lambda_max),
    )


def show_directions(directions: Sequence[tuple[float, float]]) -> str:
    """Render viewing directions the way the prompt and the -directions argument accept them."""
    return " ".join(f"{fmtnum(costheta)},{fmtnum(phi)}" for costheta, phi in directions)


def show_lambda_ranges(lambdaranges: Sequence[tuple[float, float]]) -> str:
    """Render wavelength ranges the way the prompt and the -lambdaranges argument accept them."""
    return " ".join(f"{fmtnum(lmin)},{fmtnum(lmax)}" for lmin, lmax in lambdaranges)


@dc.dataclass(frozen=True, slots=True)
class VpktField:
    """One editable setting: which attribute it sets, how to show it, and how to read it back."""

    attr: str
    prompt: str
    parse: Callable[[str], t.Any]
    show: Callable[[t.Any], str] = str


def get_editable_fields() -> list[VpktField]:
    """Return the settings offered by the interactive editor, in file order."""
    return [
        VpktField(
            "directions_costheta_phi", "Viewing directions as costheta,phi pairs", parse_directions, show_directions
        ),
        VpktField(
            "opacityexclusions",
            f"Opacity choices ({OPACITY_CHOICE_HELP})",
            parse_opacityexclusions,
            lambda values: " ".join(str(x) for x in values),
        ),
        VpktField(
            "override_tminmax", "Restrict virtual packets to a time window?", parse_bool, lambda x: "yes" if x else "no"
        ),
        VpktField("vspec_tmin_in_days", "Virtual packet time window start [days]", float, fmtnum),
        VpktField("vspec_tmax_in_days", "Virtual packet time window end [days]", float, fmtnum),
        VpktField(
            "custom_lambda_ranges",
            "Custom wavelength ranges as lambdamin,lambdamax pairs [Angstroms]",
            parse_lambda_ranges,
            show_lambda_ranges,
        ),
        VpktField(
            "override_thickcell_tau",
            "Skip virtual packets in optically thick cells?",
            parse_bool,
            lambda x: "yes" if x else "no",
        ),
        VpktField("cell_is_optically_thick_vpkt", "Cell optical depth counted as thick", float, fmtnum),
        VpktField("tau_max_vpkt", "Maximum optical depth before a virtual packet is discarded", float, fmtnum),
        VpktField("vgrid_on", "Produce a velocity grid map?", parse_bool, lambda x: "yes" if x else "no"),
        VpktField("tmin_vgrid_in_days", "Velocity grid map start [days]", float, fmtnum),
        VpktField("tmax_vgrid_in_days", "Velocity grid map end [days]", float, fmtnum),
        VpktField("vgrid_lambda_min", "Velocity grid map minimum wavelength [Angstroms]", float, fmtnum),
        VpktField("vgrid_lambda_max", "Velocity grid map maximum wavelength [Angstroms]", float, fmtnum),
    ]


def edit_config_interactively(config: VpktConfig, promptfunc: Callable[[str], str] = input) -> VpktConfig:
    """Ask for each setting in turn, keeping the current value when the reply is empty.

    The config is edited in place and returned. An invalid reply is reported and asked again, so
    the caller never has to handle a parse failure.
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
                continue
            break

    return config


def addargs(parser: argparse.ArgumentParser) -> None:
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
        "-vgrid-lambdamin", type=float, default=None, help="Velocity grid map minimum wavelength [Angstroms]"
    )
    parser.add_argument(
        "-vgrid-lambdamax", type=float, default=None, help="Velocity grid map maximum wavelength [Angstroms]"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Do not prompt for settings. Prompting is also skipped when stdin is not a terminal.",
    )
    add_outputfile_arg(parser, helptext=f"Path/filename for the output file (default {defaultoutputfile})")


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
        "vgrid_lambda_min": "vgrid_lambdamin",
        "vgrid_lambda_max": "vgrid_lambdamax",
    }
    for attr, argname in argname_of_attr.items():
        value = getattr(args, argname, None)
        if value is not None:
            setattr(config, attr, value)

    return config


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Create or interactively edit a vpkt.txt virtual packet configuration file for an ARTIS simulation."""
    args = at.parse_cli_args(addargs, main.__doc__, args, argsraw, kwargs)

    outputfile = resolve_outputfile(args.outputfile, defaultoutputfile)

    if outputfile.is_file():
        config = parse_vpkt_input(outputfile.read_text(encoding="utf-8"))
        print(f"Read existing {outputfile}")
    else:
        config = VpktConfig()
        print(f"{outputfile} does not exist, so starting from the default settings")

    config = apply_args_to_config(config, args)

    if args.non_interactive:
        pass
    elif sys.stdin.isatty():
        config = edit_config_interactively(config)
    else:
        print("stdin is not a terminal, so keeping the settings above without prompting")

    outputfile.write_text(format_vpkt_input(config), encoding="utf-8")
    at.print_saved(outputfile)


if __name__ == "__main__":
    main()
