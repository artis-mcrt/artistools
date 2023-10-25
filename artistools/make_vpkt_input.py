from pathlib import Path

Nobs = 5  # number of observer orientations

nz_obs_vpkt = "1 0.707 0 -0.707 -1"  # Cos(theta) to the observer. A list in the case of many observers

phiobs = "0 0 0 0 0"  # phi to the observer (degrees). A list in the case of many observers

nspectra_customlist_flag = "0"  # Nspectra opacity choices (i.e. Nspectra spectra for each observer)

# time window. If override_tminmax=1 it restrict vpkt to time windown
override_tminmax = 0
vspec_tmin_in_days = 0.2
vspec_tmax_in_days = 1.5

# frequency window
flag_custom_freq_ranges = 1  # restrict vpkt to a frequency range
Nrange = 1  # number of ranges
lambda_min = 3500
lambda_max = 18000  # this can be a list of ranges -- not implemented.

overrride_thickcell_tau = 1  # if overrride_thickcell_tau=1  vpkt are not created when cell optical depth is larger than cell_is_optically_thick_vpkt
cell_is_optically_thick_vpkt = 100

tau_max_vpkt = 10  # Maximum optical depth. If a vpkt reaches tau_max_vpkt it is thrown away

vgrid_on = 0  # if in_vgrid_on = 1 produce velocity grid map

# Specify time range for velocity grid map. Used if vgrid_on=1
tmin_vgrid_in_days = 0.2
tmax_vgrid_in_days = 1.5

# Specify wavelength range for velocity grid map: number of intervals (Nrange_grid) and limits (dum10,dum11)
Nrange_grid = 1
vgrid_lambda_min = 3500
vgrid_lambda_max = 6000  # can have multiple ranges -- not implemented

new_vpktfile = Path() / "vpkt.txt"
with new_vpktfile.open("w") as vpktfile:
    vpktfile.write(
        f"{Nobs}\n"
        f"{nz_obs_vpkt}\n"
        f"{phiobs}\n"
        f"{nspectra_customlist_flag}\n"
        f"{override_tminmax} {vspec_tmin_in_days} {vspec_tmax_in_days}\n"
        f"{flag_custom_freq_ranges} {Nrange} {lambda_min} {lambda_max}\n"
        f"{overrride_thickcell_tau} {cell_is_optically_thick_vpkt}\n"
        f"{tau_max_vpkt}\n"
        f"{vgrid_on}\n"
        f"{tmin_vgrid_in_days} {tmax_vgrid_in_days}\n"
        f"{Nrange_grid} {vgrid_lambda_min} {vgrid_lambda_max}"  # this can have multiple wavelength ranges. May need changed.
    )  # this can have multiple wavelength ranges. May need changed.
