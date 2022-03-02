#!/usr/bin/env python3
import argparse
import glob
import multiprocessing
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import artistools as at


def read_logfiles(modelpath):
    modeldata, _, _ = at.inputmodel.get_modeldata(modelpath)

    mpiranklist = at.get_mpiranklist(modelpath)
    nprocs = at.get_nprocs(modelpath)

    logfilepaths = []

    for folderpath in at.get_runfolders(modelpath):
        nfilesread_thisfolder = 0
        # for mpirank in range(nprocs):
        for mpirank in mpiranklist:
            logfilename = f"output_{mpirank}-0.txt"
            logfilepath = Path(folderpath, logfilename)
            if not logfilepath.is_file():
                # logfilepath = Path(folderpath, logfilename + '.gz')
                # if not logfilepath.is_file():
                #     print(f'Warning: Could not find {logfilepath.relative_to(modelpath.parent)}')
                continue
            logfilepaths.append(logfilepath)

    # print(logfilepaths)
    return logfilepaths


def read_time_taken(logfilepaths):
    logfiledict = {}

    updategrid_dict = {}
    updatepackets_dict = {}
    writeestimators_dict = {}

    for logfilepath in logfilepaths:
        mpi_process = int(str(logfilepath).split("/")[-1].split("-")[0].split("_")[-1])
        with open(logfilepath, "r") as logfile:

            lineswithtimes = [line.split(" ") for line in logfile if "took" in line]

        # for line in lineswithtimes:
        #     print(line)

        # get times for update_grid:
        updategridlines = [line for line in lineswithtimes if line[2] == "update_grid:"]
        for line in updategridlines:
            timestep = int(line[1].strip(":"))
            process = int(line[4])
            timetaken = int(line[-2])

            if timestep not in updategrid_dict:
                updategrid_dict[timestep] = {}
            if process not in updategrid_dict[timestep]:
                updategrid_dict[timestep][process] = timetaken

        # Get times for update packets
        updatepacketlines = [line for line in lineswithtimes if line[4] == "update" and line[5] == "packets"]
        # print(updatepacketlines)
        for line in updatepacketlines:
            timestep = int(line[1].strip(":"))
            process = mpi_process
            timetaken = int(line[-2])
            # print(timestep, process, timetaken)
            # if timestep == 30:
            # print(process, timetaken)

            if timestep not in updatepackets_dict:
                updatepackets_dict[timestep] = {}
            if process not in updatepackets_dict[timestep]:
                updatepackets_dict[timestep][process] = timetaken

        writetoestimatorslines = [line for line in lineswithtimes if line[0] == "writing" and line[2] == "estimators"]
        for line in writetoestimatorslines:
            timestep = int(line[7].split("...")[0])
            process = mpi_process
            timetaken = int(line[-2])
            # print(timestep, process, timetaken)
            # # if timestep == 30:
            # # print(process, timetaken)
            #
            if timestep not in writeestimators_dict:
                writeestimators_dict[timestep] = {}
            if process not in writeestimators_dict[timestep]:
                writeestimators_dict[timestep][process] = timetaken

    # print(updatepackets_dict[30])
    logfiledict["update_grid"] = updategrid_dict
    logfiledict["update_packets"] = updatepackets_dict
    logfiledict["write_estimators"] = writeestimators_dict
    # print(logfiledict['update_packets'][30])
    return logfiledict


def make_plot(logfiledict):
    for timestep in range(55):
        plotvalues = ["update_packets", "update_grid", "write_estimators"]
        for plotvalue in plotvalues:
            # print(logfiledict[plotvalue][timestep])
            process, timetaken = zip(*logfiledict[plotvalue][timestep].items())
            # print(process, timetaken)
            plt.plot(process, timetaken, label=plotvalue)
        plt.xlabel("mpi rank")
        plt.ylabel("time (s)")
        plt.title(f"timestep {timestep}")
        plt.legend()
        plt.show()

    # print(updategriddict.items())

    # for _ in modelgridindex[plotvalue][timestep]:
    #     print(_)


def addargs(parser):
    parser.add_argument(
        "-modelpath",
        default=[],
        nargs="*",
        action=at.AppendPath,
        help="Path to ARTIS model folders with model.txt and abundances.txt",
    )


def main(args=None, argsraw=None, **kwargs):
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=at.CustomArgHelpFormatter, description="Make 1D model from cone in 3D model."
        )
        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    if not args.modelpath:
        args.modelpath = [Path(".")]

    # make_plot(logfiledict)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
