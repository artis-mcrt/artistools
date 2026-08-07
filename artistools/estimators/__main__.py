"""Entry point for `python -m artistools.estimators`."""

from artistools.estimators import plotestimators


def main() -> None:
    """Plot ARTIS estimators."""
    plotestimators.main()


if __name__ == "__main__":
    main()
