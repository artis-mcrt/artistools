"""Entry point for `python -m artistools.spectra`."""

from artistools.spectra import plotspectra


def main() -> None:
    """Plot ARTIS spectra."""
    plotspectra.main()


if __name__ == "__main__":
    main()
