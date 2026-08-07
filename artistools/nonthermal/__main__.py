"""Entry point for `python -m artistools.nonthermal`."""

from artistools.nonthermal import solvespencerfanocmd


def main() -> None:
    """Solve the Spencer-Fano equation for the nonthermal electron spectrum."""
    solvespencerfanocmd.main()


if __name__ == "__main__":
    main()
