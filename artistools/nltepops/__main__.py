"""Entry point for `python -m artistools.nltepops`."""

from artistools.nltepops import plotnltepops


def main() -> None:
    """Plot ARTIS NLTE level populations."""
    plotnltepops.main()


if __name__ == "__main__":
    main()
