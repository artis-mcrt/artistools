"""Non-thermal energy deposition and Spencer-Fano equation solving."""

__all__ = ["plot", "spencerfano"]

from artistools.nonthermal import spencerfano as spencerfano
from artistools.nonthermal.spencerfano import addargs as addargs
from artistools.nonthermal.spencerfano import main as plot
