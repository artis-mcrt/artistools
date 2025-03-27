from pathlib import Path


def main() -> None:
    directions_costheta_phi = [(1, 0), (0, 0), (-1, 0)]

    # Nspectra opacity choices (i.e. Nspectra spectra for each observer)
    # 0: full opacity, -1: no line opacity; -2: no bf opacity; -3: no ff opacity; -4: no es opacity,
    # +ve: exclude element with atomic number's contribution to bound-bound opacity
    opacityexclusions: list[int] = []

    # time window. If override_tminmax=1 it restrict vpkt to time windown
    override_tminmax = 0
    vspec_tmin_in_days = 0.2
    vspec_tmax_in_days = 1.5

    custom_lambda_ranges: list[tuple[float, float]] = [(3500, 18000)]

    override_thickcell_tau = True  # if override_thickcell_tau=1  vpkt are not created when cell optical depth is larger than cell_is_optically_thick_vpkt
    cell_is_optically_thick_vpkt = 100

    # maximum optical depth before vpkt is thrown away
    tau_max_vpkt = 10

    # produce velocity grid map?
    vgrid_on = False

    # Specify time range for velocity grid map. Used if vgrid_on=1
    tmin_vgrid_in_days = 0.2
    tmax_vgrid_in_days = 1.5

    # Specify wavelength range for velocity grid map: number of intervals (Nrange_grid) and limits (dum10,dum11)
    Nrange_grid = 1
    vgrid_lambda_min = 3500
    vgrid_lambda_max = 6000  # can have multiple ranges -- not implemented

    str_custom_lambda_ranges = (
        (f" {len(custom_lambda_ranges)}" + " ".join(f"{lmin} {lmax}" for lmin, lmax in custom_lambda_ranges))
        if custom_lambda_ranges
        else ""
    )

    new_vpktfile = Path() / "vpkt.txt"
    with new_vpktfile.open("w", encoding="utf-8") as vpktfile:
        vpktfile.write(
            f"{len(directions_costheta_phi)}\n"
            f"{' '.join(str(costheta) for costheta, _ in directions_costheta_phi)}\n"
            f"{' '.join(str(phi) for _, phi in directions_costheta_phi)}\n"
            f"{bool(opacityexclusions):d} {f'{len(opacityexclusions)} ' + ' '.join(str(x) for x in opacityexclusions) if opacityexclusions else ''}\n"
            f"{override_tminmax} {vspec_tmin_in_days} {vspec_tmax_in_days}\n"
            f"{bool(custom_lambda_ranges):d}{str_custom_lambda_ranges}\n"
            f"{override_thickcell_tau:d} {cell_is_optically_thick_vpkt}\n"
            f"{tau_max_vpkt}\n"
            f"{vgrid_on:d}\n"
            f"{tmin_vgrid_in_days} {tmax_vgrid_in_days}\n"
            f"{Nrange_grid} {vgrid_lambda_min} {vgrid_lambda_max}"  # this can have multiple wavelength ranges. May need changed.
        )


if __name__ == "__main__":
    main()
