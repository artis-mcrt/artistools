"""Entry point for `python -m artistools.lightcurve`."""

from artistools.lightcurve import plotlightcurve


def main() -> None:
    """Plot ARTIS light curves."""
    plotlightcurve.main()


if __name__ == "__main__":
    main()
