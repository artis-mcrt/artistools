#!/usr/bin/env python3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

xminvalue, xmaxvalue = 3500, 7000

h = 6.62607004e-34  # m^2 kg / s
c = 299792458  # m / s


def main() -> None:
    logfiles = list(Path().glob("**/output_0-0.txt"))
    if not logfiles:
        print("no output log files found")
        sys.exit()

    fig, axes = plt.subplots(
        nrows=2, ncols=1, sharey=True, figsize=(8, 5), tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0}
    )
    assert isinstance(axes, np.ndarray)

    for index, logfilename in enumerate(logfiles):
        runfolder = logfilename.parent

        timesteptimes: list[str] = []
        with (runfolder / "light_curve.out").open(encoding="utf-8") as lcfile:
            timesteptimes.extend(line.split()[0] for line in lcfile)
        timesteptimes = timesteptimes[: len(timesteptimes) // 2]

        stats: list[dict[str, int]] = []

        with logfilename.open(encoding="utf-8") as flog:
            for line in flog:
                if line.startswith("timestep "):
                    currenttimestep = int(line.split(" ")[1].split(",")[0])
                    stats.append({})
                    if len(stats) != currenttimestep + 1:
                        print("WRONG TIMESTEP!")
                if line.startswith("k_stat_"):
                    (key, value) = line.split(" = ")
                    stats[-1][key] = int(value)

        linelabel = runfolder

        linestyle = ["-", "--"][int(index / 7)]
        yvalues = [timestepstats["k_stat_to_r_fb"] for timestepstats in stats]
        axes[0].plot(timesteptimes, yvalues, linestyle=linestyle, linewidth=1.5, label=linelabel)
        yvalues = [timestepstats["k_stat_to_ma_collexc"] for timestepstats in stats]
        axes[1].plot(timesteptimes, yvalues, linestyle=linestyle, linewidth=1.5, label=linelabel)

    for axis in axes:
        axis.set_xlim(250, 300)

    axes[0].legend(loc="best", handlelength=2, frameon=False, numpoints=1, prop={"size": 9})
    axes[-1].set_xlabel(r"Time (days)")
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))
    axes[0].set_ylabel(r"k_stat_to_r_fb")
    axes[1].set_ylabel(r"k_stat_to_ma_collexc")

    fig.savefig("plotartisstats.pdf", format="pdf")
    plt.close()


if __name__ == "__main__":
    main()
