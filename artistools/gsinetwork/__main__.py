"""Entry point for `python -m artistools.gsinetwork`."""

from artistools.gsinetwork import plotqdotabund


def main() -> None:
    """Plot ARTIS heating rates and abundances against the nuclear network trajectories."""
    plotqdotabund.main()


if __name__ == "__main__":
    main()
