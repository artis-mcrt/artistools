"""Compare ARTIS abundances and heating rates against GSI nuclear network trajectory calculations."""

__all__ = ["comparetogsinetwork", "plot"]

from artistools.gsinetwork import comparetogsinetwork as comparetogsinetwork
from artistools.gsinetwork import decayproducts as decayproducts
from artistools.gsinetwork.comparetogsinetwork import addargs as addargs
from artistools.gsinetwork.comparetogsinetwork import main as plot
