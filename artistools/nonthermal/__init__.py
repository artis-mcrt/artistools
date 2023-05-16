"""Artistools - spectra related functions."""
import artistools.nonthermal.solvespencerfanocmd

from ._nonthermal_core import analyse_ntspectrum
from ._nonthermal_core import ar_xs
from ._nonthermal_core import calculate_frac_heating
from ._nonthermal_core import calculate_Latom_excitation
from ._nonthermal_core import calculate_Latom_ionisation
from ._nonthermal_core import calculate_N_e
from ._nonthermal_core import calculate_nt_frac_excitation
from ._nonthermal_core import differentialsfmatrix_add_ionization_shell
from ._nonthermal_core import e_s_test
from ._nonthermal_core import get_arxs_array_ion
from ._nonthermal_core import get_arxs_array_shell
from ._nonthermal_core import get_electronoccupancy
from ._nonthermal_core import get_energyindex_gteq
from ._nonthermal_core import get_energyindex_lteq
from ._nonthermal_core import get_epsilon_avg
from ._nonthermal_core import get_fij_ln_en_ionisation
from ._nonthermal_core import get_J
from ._nonthermal_core import get_Latom_axelrod
from ._nonthermal_core import get_Lelec_axelrod
from ._nonthermal_core import get_lotz_xs_ionisation
from ._nonthermal_core import get_mean_binding_energy
from ._nonthermal_core import get_mean_binding_energy_alt
from ._nonthermal_core import get_nne
from ._nonthermal_core import get_nne_nt
from ._nonthermal_core import get_nnetot
from ._nonthermal_core import get_nntot
from ._nonthermal_core import get_xs_excitation
from ._nonthermal_core import get_xs_excitation_vector
from ._nonthermal_core import get_Zbar
from ._nonthermal_core import get_Zboundbar
from ._nonthermal_core import lossfunction
from ._nonthermal_core import lossfunction_axelrod
from ._nonthermal_core import namedtuple
from ._nonthermal_core import Psecondary
from ._nonthermal_core import read_binding_energies
from ._nonthermal_core import read_colliondata
from ._nonthermal_core import sfmatrix_add_excitation
from ._nonthermal_core import sfmatrix_add_ionization_shell
from ._nonthermal_core import solve_spencerfano_differentialform
from ._nonthermal_core import workfunction_tests
from .plotnonthermal import addargs
from .plotnonthermal import main
from .plotnonthermal import main as plot
