"""Create a vpkt.txt virtual packet configuration file for an ARTIS simulation."""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import artistools as at
from artistools.misc import add_outputfile_arg
from artistools.misc import resolve_outputfile

defaultoutputfile = "vpkt.txt"


# ARTIS reads these as floats, so render whole numbers without a trailing .0 to keep the file tidy
def fmtnum(value: float) -> str:
    """Format a number for vpkt.txt, dropping the decimal part when it is a whole number."""
    return str(int(value)) if value.is_integer() else str(value)


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
        msg = "At least one viewing direction is required, e.g. -directions=1,0"
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


def format_vpkt_input(
    directions_costheta_phi: Sequence[tuple[float, float]] = ((1, 0), (0, 0), (-1, 0)),
    opacityexclusions: Sequence[int] = (),
    override_tminmax: bool = False,
    vspec_tmin_in_days: float = 0.2,
    vspec_tmax_in_days: float = 1.5,
    custom_lambda_ranges: Sequence[tuple[float, float]] = (),
    override_thickcell_tau: bool = True,
    cell_is_optically_thick_vpkt: float = 100,
    tau_max_vpkt: float = 10,
    vgrid_on: bool = False,
    tmin_vgrid_in_days: float = 0.2,
    tmax_vgrid_in_days: float = 1.5,
    nrange_grid: int = 1,
    vgrid_lambda_min: float = 3500,
    vgrid_lambda_max: float = 6000,
) -> str:
    """Return the contents of a vpkt.txt file.

    opacityexclusions selects the opacity treatment for each of the Nspectra spectra per observer:
    0 for full opacity, -1 for no line opacity, -2 for no bound-free, -3 for no free-free, -4 for no
    electron scattering, and a positive atomic number to exclude that element's bound-bound opacity.
    """
    str_opacityexclusions = (
        f"{len(opacityexclusions)} " + " ".join(str(x) for x in opacityexclusions) if opacityexclusions else ""
    )
    str_custom_lambda_ranges = (
        (
            f" {len(custom_lambda_ranges)} "
            + " ".join(f"{fmtnum(lmin)} {fmtnum(lmax)}" for lmin, lmax in custom_lambda_ranges)
        )
        if custom_lambda_ranges
        else ""
    )

    return (
        f"{len(directions_costheta_phi)}\n"
        f"{' '.join(fmtnum(costheta) for costheta, _ in directions_costheta_phi)}\n"
        f"{' '.join(fmtnum(phi) for _, phi in directions_costheta_phi)}\n"
        f"{bool(opacityexclusions):d} {str_opacityexclusions}\n"
        f"{override_tminmax:d} {fmtnum(vspec_tmin_in_days)} {fmtnum(vspec_tmax_in_days)}\n"
        f"{bool(custom_lambda_ranges):d}{str_custom_lambda_ranges}\n"
        f"{override_thickcell_tau:d} {fmtnum(cell_is_optically_thick_vpkt)}\n"
        f"{fmtnum(tau_max_vpkt)}\n"
        f"{vgrid_on:d}\n"
        f"{fmtnum(tmin_vgrid_in_days)} {fmtnum(tmax_vgrid_in_days)}\n"
        # only one wavelength range is supported here, although the file format allows several
        f"{nrange_grid} {fmtnum(vgrid_lambda_min)} {fmtnum(vgrid_lambda_max)}"
    )


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-directions",
        type=parse_directions,
        default=[(1, 0), (0, 0), (-1, 0)],
        help=(
            "Viewing directions as whitespace-separated costheta,phi pairs in one quoted string."
            " Use an equals sign when the first value is negative, e.g. -directions='-1,0 0,0 1,0'"
        ),
    )
    parser.add_argument(
        "-opacityexclusions",
        type=int,
        nargs="+",
        default=[],
        help=(
            "Opacity choice per spectrum: 0 full, -1 no lines, -2 no bound-free, -3 no free-free,"
            " -4 no electron scattering, or a positive atomic number to exclude that element's lines"
        ),
    )
    parser.add_argument(
        "--override-tminmax",
        action="store_true",
        help="Restrict virtual packets to the time window given by -vspec-tmin and -vspec-tmax",
    )
    parser.add_argument("-vspec-tmin", type=float, default=0.2, help="Start of the virtual packet time window [days]")
    parser.add_argument("-vspec-tmax", type=float, default=1.5, help="End of the virtual packet time window [days]")
    parser.add_argument(
        "-lambdaranges",
        type=parse_lambda_range,
        nargs="+",
        default=[],
        help="Custom wavelength ranges as lambdamin,lambdamax pairs in Angstroms",
    )
    parser.add_argument(
        "--no-override-thickcell-tau",
        dest="override_thickcell_tau",
        action="store_false",
        help="Create virtual packets even in cells more optically thick than -cell-is-optically-thick",
    )
    parser.add_argument(
        "-cell-is-optically-thick",
        type=float,
        default=100,
        help="Cell optical depth above which virtual packets are not created",
    )
    parser.add_argument(
        "-tau-max", type=float, default=10, help="Maximum optical depth before a virtual packet is discarded"
    )
    parser.add_argument("--vgrid", action="store_true", help="Produce a velocity grid map")
    parser.add_argument("-vgrid-tmin", type=float, default=0.2, help="Start of the velocity grid map range [days]")
    parser.add_argument("-vgrid-tmax", type=float, default=1.5, help="End of the velocity grid map range [days]")
    parser.add_argument(
        "-vgrid-lambdamin", type=float, default=3500, help="Velocity grid map minimum wavelength [Angstroms]"
    )
    parser.add_argument(
        "-vgrid-lambdamax", type=float, default=6000, help="Velocity grid map maximum wavelength [Angstroms]"
    )
    add_outputfile_arg(parser, helptext=f"Path/filename for the output file (default {defaultoutputfile})")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Create a vpkt.txt virtual packet configuration file for an ARTIS simulation."""
    args = at.parse_cli_args(addargs, main.__doc__, args, argsraw, kwargs)

    outputfile = resolve_outputfile(args.outputfile, defaultoutputfile)

    vpktinput = format_vpkt_input(
        directions_costheta_phi=args.directions,
        opacityexclusions=args.opacityexclusions,
        override_tminmax=args.override_tminmax,
        vspec_tmin_in_days=args.vspec_tmin,
        vspec_tmax_in_days=args.vspec_tmax,
        custom_lambda_ranges=args.lambdaranges,
        override_thickcell_tau=args.override_thickcell_tau,
        cell_is_optically_thick_vpkt=args.cell_is_optically_thick,
        tau_max_vpkt=args.tau_max,
        vgrid_on=args.vgrid,
        tmin_vgrid_in_days=args.vgrid_tmin,
        tmax_vgrid_in_days=args.vgrid_tmax,
        vgrid_lambda_min=args.vgrid_lambdamin,
        vgrid_lambda_max=args.vgrid_lambdamax,
    )

    Path(outputfile).write_text(vpktinput, encoding="utf-8")
    at.print_saved(outputfile)


if __name__ == "__main__":
    main()
