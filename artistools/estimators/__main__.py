import multiprocessing

from .plotestimators import main as plot


def main() -> None:
    plot()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
