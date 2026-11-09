"""Physical constants and unit conversion factors, in cgs unless the name says otherwise."""

c_ang_per_s = 2.99792458e18  # Speed of light in [Angstrom/s]
C_cm_per_s = 2.99792458e10  # Speed of light in [cm/s]

h_ev_s = 4.1356677e-15  # Planck constant [eV s]
h_erg_s = 6.6260755e-27  # Planck constant [erg s]

hc_in_ev_cm = 0.0001239841984332003  # Planck constant times speed of light [eV cm]
hc_in_ev_angstrom = 12398.419843320025  # Planck constant times speed of light [eV Angstrom]

K_B_ev_per_K = 8.617333262145179e-05  # Boltzmann constant [eV / K]
K_B_erg_per_K = 1.38064852e-16  # Boltzmann constant [erg / K]

# Atomic mass unit [g] (CODATA). After artis-mcrt/artis#670, ARTIS uses this value to convert a mass number to grams.
amu_g = 1.66053906892e-24

EV_to_erg = 1.6021772e-12  # Electronvolt [erg]
MEV_to_erg = 1.6021772e-6  # Megaelectronvolt [erg]

km_to_cm = 1e5  # Kilometre [cm], also converts a velocity in km/s to cm/s
megaparsec_to_cm = 3.085677581491367e24  # Megaparsec [cm]
Msun_to_g = 1.989e33  # Solar mass [g]
Lsun_to_erg_per_s = 3.826e33  # Solar luminosity [erg/s]
Mbol_sun = 4.74  # Absolute bolometric magnitude of the Sun

day_to_s = 86400.0  # Day [s]
hour_to_s = 3600.0  # Hour [s]

angstroms_per_cm = 1e8  # Centimetre [Angstrom]
angstroms_per_micron = 1e4  # Micrometre [Angstrom]
angstroms_per_nm = 10.0  # Nanometre [Angstrom]

# m_e c / (8 pi^2 e^2): A_ul = f_lu (g_l / g_u) / (this lambda^2) with lambda in [Angstrom]
me_c_over_8pi2_e2_s_per_angstrom2 = 1.49919e-16

# The geometric units of the hydrodynamic models (G = c = Msun = 1)
geometric_time_to_s = 4.926e-6  # Time [s]
geometric_length_to_km = 1.478  # Length [km]
geometric_density_to_g_per_cm3 = 6.176e17  # Density [g/cm3]

# The refractive index of standard air is (n - 1) * 1e8 = A + B / (C - sigma^2) + D / (E - sigma^2). The vacuum
# wavenumber sigma is in [1/micron]. The tuple holds (A, B, C, D, E).
# Source: Peck & Reeder (1972), J. Opt. Soc. Am., 62, 958, doi:10.1364/JOSA.62.000958.
peck_reeder_air_coefficients = (8060.51, 2480990.0, 132.274, 17455.7, 39.32957)

# The flat cosmology of the luminosity distance of a reference light curve that gives only its redshift z. These are
# round values that many papers use, and not the result of one measurement.
H0_km_per_s_per_mpc = 70.0
omega_matter = 0.3
