"""Artistools - spectra related functions."""

from artistools.nonthermal import solvespencerfanocmd
from artistools.nonthermal._nonthermal_core import analyse_ntspectrum
from artistools.nonthermal._nonthermal_core import ar_xs
from artistools.nonthermal._nonthermal_core import calculate_frac_heating
from artistools.nonthermal._nonthermal_core import calculate_Latom_excitation
from artistools.nonthermal._nonthermal_core import calculate_Latom_ionisation
from artistools.nonthermal._nonthermal_core import calculate_N_e
from artistools.nonthermal._nonthermal_core import calculate_nt_frac_excitation
from artistools.nonthermal._nonthermal_core import differentialsfmatrix_add_ionization_shell
from artistools.nonthermal._nonthermal_core import e_s_test
from artistools.nonthermal._nonthermal_core import get_arxs_array_ion
from artistools.nonthermal._nonthermal_core import get_arxs_array_shell
from artistools.nonthermal._nonthermal_core import get_electronoccupancy
from artistools.nonthermal._nonthermal_core import get_energyindex_gteq
from artistools.nonthermal._nonthermal_core import get_energyindex_lteq
from artistools.nonthermal._nonthermal_core import get_epsilon_avg
from artistools.nonthermal._nonthermal_core import get_fij_ln_en_ionisation
from artistools.nonthermal._nonthermal_core import get_J
from artistools.nonthermal._nonthermal_core import get_Latom_axelrod
from artistools.nonthermal._nonthermal_core import get_Lelec_axelrod
from artistools.nonthermal._nonthermal_core import get_lotz_xs_ionisation
from artistools.nonthermal._nonthermal_core import get_mean_binding_energy
from artistools.nonthermal._nonthermal_core import get_mean_binding_energy_alt
from artistools.nonthermal._nonthermal_core import get_nne
from artistools.nonthermal._nonthermal_core import get_nne_nt
from artistools.nonthermal._nonthermal_core import get_nnetot
from artistools.nonthermal._nonthermal_core import get_nntot
from artistools.nonthermal._nonthermal_core import get_xs_excitation
from artistools.nonthermal._nonthermal_core import get_xs_excitation_vector
from artistools.nonthermal._nonthermal_core import get_Zbar
from artistools.nonthermal._nonthermal_core import get_Zboundbar
from artistools.nonthermal._nonthermal_core import lossfunction
from artistools.nonthermal._nonthermal_core import lossfunction_axelrod
from artistools.nonthermal._nonthermal_core import namedtuple
from artistools.nonthermal._nonthermal_core import Psecondary
from artistools.nonthermal._nonthermal_core import read_binding_energies
from artistools.nonthermal._nonthermal_core import read_colliondata
from artistools.nonthermal._nonthermal_core import sfmatrix_add_excitation
from artistools.nonthermal._nonthermal_core import sfmatrix_add_ionization_shell
from artistools.nonthermal._nonthermal_core import solve_spencerfano_differentialform
from artistools.nonthermal._nonthermal_core import workfunction_tests
from artistools.nonthermal.plotnonthermal import addargs
from artistools.nonthermal.plotnonthermal import main as plot
