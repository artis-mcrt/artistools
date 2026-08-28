"""Entry point for `python -m artistools.nonthermal`."""

from importlib.util import find_spec

from artistools.commands import run_module_as_subcommand

if __name__ == "__main__":
    run_module_as_subcommand(find_spec("artistools.nonthermal.solvespencerfanocmd"))
