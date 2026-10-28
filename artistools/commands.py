"""Define the artistools subcommand tree and dispatch to each subcommand's module."""

import argparse
import dataclasses as dc
import importlib
import sys
import typing as t
from collections.abc import Mapping
from functools import lru_cache
from types import MappingProxyType

from artistools.misc import addarg_common
from artistools.misc import CustomArgHelpFormatter
from artistools.misc import get_shown_flags
from artistools.misc import print_product
from artistools.misc import SuggestingArgumentParser
from artistools.paths import get_path

if t.TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Sequence


def get_example_specs() -> "tuple[tuple[str, str, CommandSpec], ...]":
    """Return every example of the tree as (command line, description, spec of the command), for the test.

    The help lists no hidden command, thus its examples stay out. A hidden name such as describeinputmodel then does
    not repeat the examples of its visible name.
    """

    def walk(tree: CommandTree, path: str) -> "Generator[tuple[str, str, CommandSpec]]":
        for name, spec in tree.items():
            if isinstance(spec, dict):
                yield from walk(spec, f"{path}{name} ")
            elif not spec.hidden:
                yield from (
                    (f"{path}{name} {arguments}".rstrip(), description, spec)
                    for arguments, description in spec.examples
                )

    return tuple(walk(subcommandtree, ""))


def get_examples(tree: "CommandTree | None" = None, path: str = "") -> tuple[tuple[str, str], ...]:
    """Return the examples of the commands of one level of the tree as (command line, description).

    The help of a command group, e.g. inputmodel, lists the examples of its own commands. The top-level help thus
    names no command of a group, and it stays short. path holds the names of the groups above the tree.
    """
    level = subcommandtree if tree is None else tree
    return tuple(
        (f"{path}{name} {arguments}".rstrip(), description)
        for name, spec in level.items()
        if isinstance(spec, CommandSpec) and not spec.hidden
        for arguments, description in spec.examples
    )


# the help aligns the descriptions of the examples at this width of a command line
MAXEXAMPLEWIDTH = 40


def format_examples(examples: "Sequence[tuple[str, str]]") -> str:
    """Return the block of examples for a help text, with the descriptions in one column."""
    width = min(max(len(command) for command, _ in examples), MAXEXAMPLEWIDTH)
    lines = ["Examples (a path of . reads the model in the working folder):"]
    for command, description in examples:
        if len(command) <= width:
            lines.append(f"  artistools {command:{width}}  # {description}")
        else:
            # a long command line takes its description to the next line, thus the column of the descriptions stays
            lines.extend((f"  artistools {command}", f"  {'':{len('artistools ') + width}}  # {description}"))

    return "\n".join(lines)


def get_command_epilog(words: "Sequence[str]", spec: "CommandSpec") -> str | None:
    """Return the examples of one command for its own help, or None when it has none.

    words is the full name of the command, e.g. ("inputmodel", "describe"), thus each example runs as it is.
    """
    if not spec.examples:
        return None

    return format_examples([
        (" ".join([*words, arguments]).rstrip(), description) for arguments, description in spec.examples
    ])


def get_group_epilog(words: "Sequence[str]", tree: "CommandTree") -> str:
    """Return the examples of the commands of one group and the pointer to the help of one command."""
    pointer = f"Run `artistools {' '.join(words)} <command> --help` for the arguments of one command."
    examples = get_examples(tree, f"{' '.join(words)} ")

    return f"{format_examples(examples)}\n\n{pointer}" if examples else pointer


def get_epilog() -> str:
    """Return the examples and the pointer to the help of one command."""
    return (
        format_examples(get_examples())
        + "\n\nRun `artistools <command> --help` for the arguments of one command."
        + "\nA command that reads a model takes its folder with --modelpath. Most of these commands also take the"
        + "\nfolder as the last positional argument."
        + "\nA time or a wavelength can have a unit, e.g. -t 12h or --lambdamin 600nm. A value with no unit"
        + "\nis in days or in Angstroms. --explain prints what the arguments select. Then it stops before it"
        + "\nreads the data."
        + "\nSet ARTISTOOLS_TRACEBACK=1 to get the full traceback of an error."
        + "\nThe exit status is 0 for success, 2 for an argument that the parser refuses, and 1 for a different"
        + "\nerror. `artistools check` also gives 1 for a check with a warning or an error."
    )


# the console scripts that take a subcommand as their first argument. Every other console script names
# its subcommand in the CommandSpec of that subcommand, thus get_script_subcommands finds it
DISPATCHERSCRIPTS = ("at", "artistools")


@dc.dataclass(frozen=True, slots=True)
class CommandSpec:
    """A subcommand definition: the implementing module and the static help text shown in command listings.

    The --help listing leaves out a hidden command, but the command still works. Examples are a deprecated duplicate
    name and the server.
    """

    module: str
    funcname: str = "main"
    helptext: str = ""
    aliases: tuple[str, ...] = ()
    hidden: bool = False

    script: str = ""
    """The per-command console script that runs this subcommand, e.g. plotartisestimators."""

    note: str = ""
    """Extra text for the help of the command alone. The listing of one line has no room for it."""

    examples: tuple[tuple[str, str], ...] = ()
    """Example invocations as (arguments, description) pairs. A test runs each one against the test
    model, thus an example cannot name an argument that a later commit takes away. A path of "."
    reads the model in the working folder."""

    exampletestfolder: str | None = "testmodel"
    """The folder below tests/data that the test of the examples gives in place of ".". None makes the test parse
    the examples and not run them. Use None only for a command that needs data that the test data does not have."""

    exampleneeds: tuple[str, ...] = ()
    """The optional packages that the examples need, e.g. plotly. A test environment with no such package parses the
    examples and does not run them, because the extras are optional."""

    def get_description(self) -> str:
        """Return the description of the help of the command: the help text and the note."""
        return f"{self.helptext} {self.note}".strip()


type CommandTree = dict[str, CommandSpec | CommandTree]

# The --help listing shows the top-level commands under these headings, in this order. A command that no
# tuple names appears under the last heading, thus a new command is still listed.
COMMANDGROUPS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "Plot commands": (
        "comparetogsinetwork",
        "plotdensity",
        "plotestimators",
        "plotinitialcomposition",
        "plotlastinteractionvelocity",
        "plotlastpacketinteraction",
        "plotlightcurves",
        "plotlinecontributions",
        "plotlinefluxes",
        "plotlogfiles",
        "plotnltepops",
        "plotopacity",
        "plotradfield",
        "plotspectra",
        "plotspherical",
        "plottransitions",
        "plotviewingangles",
    ),
    "Model commands": ("inputmodel", "makevpktinput"),
    "Data commands": (
        "cache",
        "deposition",
        "ejectaopacity",
        "exportmassfractions",
        "filterpackets",
        "gsinetworkdecayproducts",
        "hesma",
        "linecontributions",
        "spencerfano",
        "writebollightcurvedata",
        "writecodecomparisondata",
        "writespectra",
    ),
    "Other commands": (
        "catalog",
        "check",
        "completions",
        "corehours",
        "describe",
        "getpath",
        "help",
        "installskill",
        "logsummary",
        "python",
        "report",
        "timesteps",
        "values",
        "version",
    ),
})

# "artistools describeinputmodel" was a top-level command, thus a script of a user holds that name.
# One spec gives every name of this command, thus no copy of its help text can drift
DESCRIBEINPUTMODEL = CommandSpec(
    "inputmodel.describeinputmodel",
    helptext="Describe an ARTIS input model, such as the mass, velocity structure, and abundances.",
    aliases=("describeinputmodel",),
    examples=((".", "the mass, the velocity structure, and the abundances of the input model"),),
)

subcommandtree: CommandTree = {
    "cache": CommandSpec(
        "cache",
        helptext="Make, list, or delete the parquet caches of the text output of an ARTIS model.",
        note=(
            "A reader converts a text file to a cache at the first read. 'make' does all the conversions at one"
            " time, e.g. in the job script after a run of ARTIS."
        ),
        examples=(
            ("make .", "the caches of the input model, the line list, the estimators, and the packets"),
            ("list .", "the caches of the model and their sizes"),
        ),
    ),
    "catalog": CommandSpec(
        "catalog",
        helptext="Make a table of the ARTIS models below a folder, with one row for each model.",
        note=(
            "The table gives the cells, the time steps, the main physics options, the ARTIS version, and the peak"
            " luminosity. It reads no packets, estimators, or spectra."
        ),
        examples=((".", "the table of the models below the working folder"),),
    ),
    "check": CommandSpec(
        "check",
        helptext="Check whether the results of an ARTIS model deserve trust, and give one row for each check.",
        aliases=("doctor",),
        note=(
            "The checks also examine the input files, the caches, and the range of the arrival times. --frompackets"
            " also reads the escaped packets to find the Monte Carlo noise of the light curve. The exit status is 1"
            " when a check gives a warning or an error, and --exit-zero gives 0."
        ),
        examples=((".", "the checks of the results of the model"),),
    ),
    "comparetogsinetwork": CommandSpec(
        "gsinetwork.comparetogsinetwork",
        helptext="Compare ARTIS to a GSI Network calculation.",
        note="The comparison covers the energy release and the abundances.",
        examples=(("--modelpath .", "the energy release and the abundances of ARTIS and the GSI Network"),),
        exampletestfolder="test-classicmode_3d",
    ),
    "completions": CommandSpec(
        "completions",
        helptext="Give the instructions for tab completion.",
        note="With a shell name, e.g. zsh, the command prints the code that the instructions write to a file.",
        examples=(("zsh", "the code of tab completion for zsh"),),
    ),
    "corehours": CommandSpec(
        "logs.corehours",
        helptext="Sum the core hours of the runs of ARTIS models from the rank-0 logs of sn3d.",
        note=(
            "A run that stopped early gets an estimate from the timestamps of its log. The sum counts a copy of a log"
            " in the model folder one time."
        ),
        examples=(
            (".", "the core hours of each run and the sums"),
            (". --output-format json", "the same values as JSON"),
        ),
        exampletestfolder="test-classicmode_3d",
    ),
    "deposition": CommandSpec(
        "estimators.deposition",
        helptext="Give the deposition rate per unit volume, per ion, and per unit mass.",
        note=(
            "The deposition_ estimators of each cell give the rate, thus the command needs the output of a"
            " recent ARTIS version. Every column covers the cells that hold matter."
        ),
        examples=(
            (". -t 300", "the deposition rates at 300 days"),
            (". --listchannels", "the deposition channels of the model"),
        ),
        # the estimators of the test models hold no deposition_ column
        exampletestfolder=None,
    ),
    "describe": CommandSpec(
        "describemodel",
        helptext="Summarise an ARTIS model, e.g. its cells, timesteps, output files, ions, and physics options.",
        note="Run it first to choose the readers and the timesteps for a question about the model.",
        examples=((".", "the summary of the model in the working folder"),),
    ),
    # the help lists this name under "inputmodel describe", thus the top level hides it
    "describeinputmodel": dc.replace(DESCRIBEINPUTMODEL, aliases=(), hidden=True),
    "ejectaopacity": CommandSpec(
        "lines.ejectaopacity",
        helptext="Compute the opacities of the ejecta.",
        note="This gives the binned expansion opacities and the Planck-mean opacities.",
        examples=(("--modelpath . -t 300", "the opacities of the ejecta at 300 days"),),
    ),
    "exportmassfractions": CommandSpec(
        "estimators.exportmassfractions",
        helptext="Write the mass fractions of the elements.",
        note="The values come from the estimators, and the command writes a text file.",
        examples=(("--modelpath . --ts 40 --modelgridindex 0", "the mass fractions of cell 0 at timestep 40"),),
    ),
    "filterpackets": CommandSpec(
        "packets.filterpackets",
        helptext="Remove each packet except the escaped r-packets from the packets files of a model.",
        note=(
            "The filter makes the files smaller. It also removes the escaped gamma packets. Each filtered file"
            " replaces its packets file as a zstd file."
        ),
        examples=((". --dry-run", "the number of packets and the size that the filter removes"),),
        exampletestfolder="test-classicmode_3d",
    ),
    "getpath": CommandSpec(
        "commands",
        funcname="get_artistools_path",
        helptext="Print the folder of the package.",
        examples=(("", "the folder of the artistools package"),),
    ),
    "gsinetworkdecayproducts": CommandSpec(
        "gsinetwork.decayproducts",
        helptext="Read the beta-decay energy of a trajectory.",
        note="The data comes from the trajectories of a nucleosynthesis calculation.",
        examples=(("--trajectoryroot trajectories", "the beta-decay energy of the trajectories in a folder"),),
        # the command writes a cache of each trajectory into the trajectory folder, thus a run changes the test data
        exampletestfolder=None,
    ),
    "help": CommandSpec(
        "help",
        helptext="Print the help of artistools or of one command.",
        note="The command prints the same text as --help, e.g. of plotspectra or of inputmodel describe.",
        examples=(("plotspectra", "the help of plotspectra, as plotspectra --help gives it"),),
    ),
    "hesma": CommandSpec(
        "hesma_scripts",
        helptext="Convert ARTIS output to the HESMA formats.",
        note="The HESMA model archive takes these file formats.",
        examples=(
            (
                "bollightcurve --modelpath . --timemin 250 --timemax 300",
                "the bolometric light curve from 250 to 300 days in the HESMA format",
            ),
        ),
    ),
    "inputmodel": {
        "describe": DESCRIBEINPUTMODEL,
        "energyfiles": CommandSpec(
            "inputmodel.energyinputfiles",
            helptext="Plot and inspect the ARTIS energy input files.",
            examples=(("plotrate --modelpath .", "the analytic nuclear heating power against time"),),
        ),
        "from_e2e": CommandSpec(
            "inputmodel.from_e2e_model",
            helptext="Prepare data for an ARTIS kilonova calculation from end-to-end hydro models.",
            examples=(("--npz model.npz -o artismodel", "an ARTIS model from the end-to-end model in model.npz"),),
            # the test data has no npz file of an end-to-end model
            exampletestfolder=None,
        ),
        "fromcmfgen": CommandSpec(
            "inputmodel.fromcmfgen.convert_to_artis",
            helptext="Convert a CMFGEN SN_HYDRO_DATA snapshot to an ARTIS model at the snapshot's own time.",
            examples=(
                (
                    "--snapshot SN_HYDRO_DATA_1.300d -o artismodel",
                    "an ARTIS model from the CMFGEN snapshot at 1.3 days",
                ),
            ),
            # the test data has no CMFGEN snapshot
            exampletestfolder=None,
        ),
        "make1dslicefrom3dmodel": CommandSpec(
            "inputmodel.make1dslicefrom3d",
            helptext="Convert abundances.txt and model.txt from a 3D model to a one-dimensional slice.",
            examples=((". --axis x -o 1dslice", "a 1D slice of the 3D model along x, in the folder 1dslice"),),
            exampletestfolder="testmodel_3d_10^3",
        ),
        "makeartismodel": CommandSpec(
            "inputmodel.makeartismodel",
            helptext="Tools to create an ARTIS input model.",
            examples=(("--modelpath . --dimensionreduce 1", "a spherically symmetric 1D model from a 3D model"),),
            exampletestfolder="test-classicmode_3d",
        ),
        "makeartismodel1dslicefromcone": CommandSpec(
            "inputmodel.slice1dfromconein3dmodel",
            helptext="Make a 1D model from a cone in a 3D model.",
            examples=(
                (
                    "--modelpath . --axis +z --coneangle 60 --nshells 4",
                    "a 1D model of 4 shells from a cone of 60 degrees around the +z axis",
                ),
            ),
            exampletestfolder="testmodel_3d_10^3",
        ),
        "makeartismodelfromparticlegridmap": CommandSpec(
            "inputmodel.modelfromhydro",
            helptext="Create an ARTIS format model from grid.dat.",
            examples=(("-i . -d 1", "a 1D ARTIS model from the grid.dat in the working folder"),),
            exampletestfolder="kilonova",
        ),
        "makeartismodelfromshen2018": CommandSpec(
            "inputmodel.shen2018",
            helptext="Convert Shen et al. 2018 models to ARTIS format.",
            examples=(("--inputpath 1.00_5050.dat", "an ARTIS model from the file 1.00_5050.dat"),),
            # the test data has no model of this format
            exampletestfolder=None,
        ),
        "makeartismodelfromsingletrajectory": CommandSpec(
            "inputmodel.rprocess_from_trajectory",
            helptext="Create an ARTIS model from single trajectory abundances.",
            examples=(
                (
                    "--trajectoryroot trajectories --particleid 133371",
                    "a model of one cell from the abundances of the particle 133371",
                ),
            ),
            # the trajectories of the test data have no abundances at the time of the model
            exampletestfolder=None,
        ),
        "maptogrid": CommandSpec(
            "inputmodel.maptogrid",
            helptext="Map tracer particle trajectories to a Cartesian grid.",
            examples=(("-i . --ncoordgrid 50", "a grid of 50 cells on each axis from ejectasnapshot.dat"),),
            exampletestfolder="kilonova",
        ),
        "opacityfile": CommandSpec(
            "inputmodel.opacityinputfile",
            helptext="Write or inspect an ARTIS grey opacity.txt.",
            examples=(("uniform --modelpath . --kappa 0.1", "an opacity.txt with 0.1 cm^2/g in each cell"),),
        ),
        "plotinitialabundances": CommandSpec(
            "inputmodel.plotinitialabundances",
            helptext="Plot initial abundances or mass fractions from one or more ARTIS models.",
            examples=((".", "the initial mass fractions against the mass number"),),
        ),
        "to_tardis": CommandSpec(
            "inputmodel.to_tardis",
            helptext="Convert an ARTIS format model to TARDIS format.",
            examples=((".", "the TARDIS model file of the input model"),),
        ),
    },
    "installskill": CommandSpec(
        "installskill",
        helptext="Install the artis-data skill for a large language model (LLM) agent, e.g. Claude Code.",
        note="The skill tells the agent how to read the data of a model with artistools.",
        examples=(("", "the skill for Claude Code, in ~/.claude/skills"),),
        # a run writes into the home folder of the user
        exampletestfolder=None,
    ),
    "linecontributions": CommandSpec(
        "lines.linecontributions",
        helptext="Give the Sobolev optical depth and the emission of each line in a wavelength range.",
        note=(
            "The level populations of each cell follow calculate_levelpop() of ARTIS. --percell gives one row for each"
            " cell and line. Use it to find where a line forms."
        ),
        examples=(
            (
                ". -t 259 --lambdamin 4000 --lambdamax 5500",
                "the lines with the most emission from 4000 to 5500 Å at 259 days",
            ),
        ),
    ),
    "logsummary": CommandSpec(
        "logs.logsummary",
        helptext="Count the warnings and the errors in the logs of an ARTIS model, by kind of message.",
        note="Each rank logs the cells that it updates, thus --ranks and --modelgridindex select the logs to read.",
        examples=((". --ranks all", "the warnings and the errors in the logs of all the ranks"),),
        exampletestfolder="test-classicmode_3d",
    ),
    "makevpktinput": CommandSpec(
        "make_vpkt_input",
        helptext="Write a vpkt.txt for a model.",
        note="The file holds the configuration of the virtual packets.",
        examples=(("--non-interactive", "a vpkt.txt with the default settings, with no prompt"),),
    ),
    "plotdensity": CommandSpec(
        "inputmodel.plotdensity",
        helptext="Plot the enclosed mass and the mass per unit velocity against velocity.",
        examples=((".", "the mass profile of the input model"),),
    ),
    "plotestimators": CommandSpec(
        "estimators.plotestimators",
        script="plotartisestimators",
        helptext="Plot ARTIS estimators.",
        examples=(
            ("Te TR . -t 300", "two estimator variables against velocity"),
            ("Te TR . -t 300 --dim 2", "each variable as a colour image at each cylindrical radius and z"),
            (". --listvariables", "every variable that a model holds"),
        ),
        aliases=("estimators",),
    ),
    "plotinitialcomposition": CommandSpec(
        "inputmodel.plotinitialcomposition",
        helptext="Plot ARTIS input model composition.",
        examples=(("rho Fe .", "the density and the mass fraction of Fe in a slice of a 3D model"),),
        exampletestfolder="test-classicmode_3d",
    ),
    "plotlastinteractionvelocity": CommandSpec(
        "packets.plotlastinteractionvelocity",
        helptext="Plot the velocity of the last interaction of the escaped packets in a wavelength range against time.",
        note=(
            "In a range with no strong line, the median is an estimate of the velocity of the photosphere. Give it to"
            " plotlinecontributions --photospherevelocity."
        ),
        examples=(
            (". --lambdamin 7000 --lambdamax 8000", "the median and the quartiles for all the directions"),
            (
                ". --lambdamin 7000 --lambdamax 8000 --dirbin 0 50 90 --average-over-phi-angle",
                "the same for the two poles and the costheta bin above the equator",
            ),
        ),
        exampletestfolder="test-classicmode_3d",
    ),
    "plotlastpacketinteraction": CommandSpec(
        "packets.plotlastpacketinteraction",
        helptext="Plot the last interaction of a packet.",
        note=(
            "The plot gives the properties of that interaction against the velocity of the ejecta. --groupby gives the"
            " luminosity of each element, ion, or line against the velocity, and plotlinecontributions gives the same"
            " view from the level populations."
        ),
        examples=(
            (". -t 6.5", "the last interactions of the packets that arrive at 6.5 days"),
            (
                ". -t 6.5 --groupby ion --lambdamin 3500 --lambdamax 4500",
                "the luminosity of each ion against the velocity, for the packets from 3500 to 4500 Å",
            ),
        ),
        exampletestfolder="test-classicmode_3d",
    ),
    "plotlightcurves": CommandSpec(
        "lightcurve.plotlightcurve",
        script="plotartislightcurve",
        helptext="Plot ARTIS light curves.",
        examples=((".", "the light curve of a model"),),
        aliases=("lc", "plotlightcurve"),
    ),
    "plotlinecontributions": CommandSpec(
        "lines.plotlinecontributions",
        helptext="Plot the Sobolev emission and the Sobolev optical depth of the strongest lines against velocity.",
        note=(
            "The values come from the estimators and the level populations, as in linecontributions."
            " plotlastpacketinteraction --groupby gives the same view from the packets."
        ),
        examples=(
            (
                ". --timedays 7 --lambdamin 3000 --lambdamax 10000",
                "where the 8 lines with the most emission from 3000 to 10000 Å form at 7 days",
            ),
            (
                ". --timedays 7 --lambdamin 6000 --lambdamax 6400 --rankby tau",
                "where the lines with the largest tau_sobolev from 6000 to 6400 Å form at 7 days",
            ),
            (
                (
                    ". --timedays 7 --lambdamin 6000 --lambdamax 6400 --rankby tau --troughdirbin 90"
                    " --average-over-phi-angle --photospherevelocity 15000km/s"
                ),
                "the same, with a Sobolev estimate of the trough for the observer at the pole",
            ),
        ),
        exampletestfolder="test-classicmode_3d",
    ),
    "plotlinefluxes": CommandSpec(
        "plotlinefluxes",
        helptext="Plot the ratios of the line fluxes.",
        note="The ratios serve a comparison to Floers.",
        examples=(("--modelpath .", "the fluxes of the default emission features against time"),),
        exampletestfolder="test-classicmode_3d",
    ),
    "plotlogfiles": CommandSpec(
        "logs.plotlogfiles",
        helptext="Plot the time that each rank took.",
        note="The times come from the log files of each run of a model.",
        examples=(("--modelpath .", "the time that each rank took in each timestep"),),
        exampletestfolder="test-classicmode_3d",
    ),
    "plotnltepops": CommandSpec(
        "nltepops.plotnltepops",
        helptext="Plot ARTIS non-LTE populations.",
        examples=(("--modelpath . -t 300 --modelgridindex 0", "the level populations of one cell"),),
        note=(
            "Give a time with --timedays or --timestep. A model of more than one cell also needs a cell,"
            " which --modelgridindex or --velocity gives."
        ),
    ),
    "plotopacity": CommandSpec(
        "lines.plotopacity",
        helptext="Plot the binned opacities of the ejecta against wavelength.",
        note=(
            "The plot shows the expansion opacity, the line-binned opacity, and the line-binned opacity with"
            " each Sobolev optical depth limited to 1. Without --cell, it shows the mean over all cells, weighted"
            " by mass. --vmin, --vmax, --dirbin, --costhetamin, and --costhetamax select the cells, e.g. the cells"
            " in front of the photosphere of one direction."
        ),
        examples=(
            (
                "--modelpath . --ts 40 --xmin 3000 --xmax 4000",
                "the opacities at timestep 40, from 3000 to 4000 Angstroms",
            ),
        ),
    ),
    "plotradfield": CommandSpec(
        "plotradfield",
        helptext="Plot the radiation field estimators.",
        examples=(("--modelpath . -t 300 --modelgridindex 0", "the radiation field of cell 0 at 300 days"),),
    ),
    "plotspectra": CommandSpec(
        "spectra.plotspectra",
        script="plotartisspectrum",
        helptext="Plot spectra from ARTIS and reference data.",
        examples=((". -t 300", "the spectrum at 300 days"),),
        aliases=("spec",),
    ),
    "plotspherical": CommandSpec(
        "plotspherical",
        helptext="Plot direction maps based on escaped packets.",
        examples=(("--modelpath . --ts 40", "the direction maps of the escaped packets at timestep 40"),),
    ),
    "plottransitions": CommandSpec(
        "plottransitions",
        helptext="Plot the spectrum of the transitions.",
        note="The spectrum comes from the bound-bound transitions.",
        examples=(("--modelpath . -t 300", "the spectrum of the transitions of cell 0 at 300 days"),),
    ),
    "plotviewingangles": CommandSpec(
        "plotviewingangles",
        helptext="Plot a 3D view of a model.",
        note="The view holds an isosurface of the density and the direction bins.",
        examples=((".", "the 3D view of the density and the direction bins"),),
        exampletestfolder="test-classicmode_3d",
        exampleneeds=("plotly",),
    ),
    "python": CommandSpec(
        "commands",
        funcname="show_python",
        helptext="Print the path of the Python interpreter that runs artistools.",
        note="A script that imports artistools can run with this interpreter, e.g. from an agent.",
        examples=(("", "the path of the Python interpreter of artistools"),),
    ),
    "report": CommandSpec(
        "report",
        helptext="Make one HTML page that is the first page to read about an ARTIS model.",
        note=(
            "The page holds the provenance, the health check, the model summary, the log summary, and standard figures."
        ),
        examples=((". -o myreport", "the report of the model in the working folder, in the folder myreport"),),
    ),
    "server": CommandSpec(
        "misc.remote",
        helptext="Run the readers of a command for a model on this host.",
        note=(
            "A command on a different host starts this server through ssh when it gets a model path of the"
            " form host:path, e.g. vae26:~/scratch/mymodel. Do not run it yourself."
        ),
        # a user does not run the server, thus the listing of the commands leaves it out
        hidden=True,
    ),
    "spencerfano": CommandSpec(
        "nonthermal.spencerfano",
        helptext="Solve the Spencer-Fano equation for a cell.",
        note="The data comes from one cell of an ARTIS model at one timestep.",
        examples=(("--modelpath . -t 300 --modelgridindex 0", "the non-thermal solution for cell 0 at 300 days"),),
    ),
    "timesteps": CommandSpec(
        "timesteps",
        helptext="List the timesteps and their days.",
        note="The table gives the days that each timestep covers.",
        examples=(
            ("--modelpath .", "the timesteps of the model and their days"),
            ("--modelpath . -t 300", "the timestep that covers 300 days"),
        ),
    ),
    "values": CommandSpec(
        "values",
        helptext="Print the raw estimator values of cells and timesteps, with no plot and no average.",
        note=(
            "The variables come first and the model path last. Each row is one cell at one timestep, and the name of"
            " each column gives its unit."
        ),
        examples=(("Te nne -t 300 --modelgridindex 0 .", "Te and nne of cell 0 at 300 days"),),
    ),
    "version": CommandSpec(
        "commands",
        funcname="show_version",
        helptext="Print the artistools version.",
        examples=(("", "the version of artistools"),),
    ),
    "writebollightcurvedata": CommandSpec(
        "lightcurve.writebollightcurvedata",
        helptext="Write the bolometric light curve of a model.",
        note="The command writes a plain text file.",
        examples=((".", "the bolometric light curve of the model as a text file"),),
    ),
    "writecodecomparisondata": CommandSpec(
        "writecomparisondata",
        helptext="Write the model in the comparison format.",
        note="The code comparison workshop takes this format.",
        examples=(("--modelpath . --selected-timesteps 50", "the files of the code comparison at timestep 50"),),
    ),
    "writespectra": CommandSpec(
        "spectra.writespectra",
        helptext="Write the spectrum of each timestep.",
        note="The command writes one text file for each timestep.",
        examples=(("--modelpath .", "one spectrum file for each timestep of the model"),),
    ),
}


class CommandGroupHeading(argparse.Action):
    """A pseudo action that carries a heading into the help listing of the top-level commands."""


def group_subactions(subactions: "list[argparse.Action]") -> "dict[str, list[argparse.Action]] | None":
    """Return the top-level subcommands keyed by the heading of COMMANDGROUPS.

    A command that no tuple of COMMANDGROUPS names goes under the last heading, thus a new command still
    appears and the headings stay. Return None for the commands of a group such as "artistools inputmodel",
    which keeps one flat listing. Such a group holds a command that is not a top-level command, e.g. energyfiles.
    """
    groupofcommand = {name: heading for heading, names in COMMANDGROUPS.items() for name in names}
    if any(sub.dest not in subcommandtree for sub in subactions):
        return None

    lastheading = list(COMMANDGROUPS)[-1]
    grouped: dict[str, list[argparse.Action]] = {heading: [] for heading in COMMANDGROUPS}
    for sub in subactions:
        grouped[groupofcommand.get(sub.dest, lastheading)].append(sub)

    return {heading: members for heading, members in grouped.items() if members}


# what a group of subcommands does, for the listing of the commands above it
GROUPHELPTEXT = {"inputmodel": "Make and inspect an ARTIS input model."}

# the heading of the commands of a group that COMMANDGROUPS does not divide, e.g. inputmodel
FLATGROUPHEADING = "Commands"


def shorten_aliases(metavar: str) -> str:
    """Return the name of a command with one alias alone, as the listing shows it.

    argparse names every alias, thus "plotlightcurves (lc, plotlightcurve)" took 36 columns of the
    listing and left 38 for the text, which then wrapped over three lines. Every alias still works, and
    the help of the command names them all.
    """
    name, _, aliases = metavar.partition(" (")
    if not aliases:
        return metavar

    return f"{name} ({aliases.rstrip(')').split(', ')[0]})"


class CommandListFormatter(CustomArgHelpFormatter):
    """Give the help of a parser that holds commands, with the commands under the headings of COMMANDGROUPS."""

    @t.override
    def _format_action(self, action: argparse.Action) -> str:
        """Render a group heading as its own line, and every other action as usual.

        The headings of the commands replace the line of the subcommand metavar, which has no help.
        """
        if isinstance(action, CommandGroupHeading):
            # Python 3.14 colours the headings of argparse with this theme. Python 3.13 has no theme
            theme = getattr(self, "_theme", None)
            return f"\n{theme.heading}{action.dest}{theme.reset}\n" if theme is not None else f"\n{action.dest}\n"

        if isinstance(action, argparse._SubParsersAction):  # ruff:ignore[private-member-access]
            return "".join(self._format_action(subaction) for subaction in self._iter_indented_subactions(action))

        return super()._format_action(action)

    @t.override
    def _iter_indented_subactions(self, action: argparse.Action) -> "Generator[argparse.Action]":
        """Yield the subcommands under a heading for each group.

        This hook also feeds the column width, thus the headings do not disturb the alignment. A heading
        replaces the line of the metavar, thus the commands have the indent of the options.
        """
        if not isinstance(action, argparse._SubParsersAction):  # ruff:ignore[private-member-access]
            yield from super()._iter_indented_subactions(action)
            return

        subactions = list(action._get_subactions())  # ruff:ignore[private-member-access]
        for heading, members in (group_subactions(subactions) or {FLATGROUPHEADING: subactions}).items():
            yield CommandGroupHeading(option_strings=[], dest=f"{heading}:", help=None)
            for member in members:
                member.metavar = shorten_aliases(str(member.metavar or member.dest))
                yield member


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add no command-line arguments (for subcommands that take none)."""


@lru_cache(maxsize=1)
def get_script_subcommands() -> Mapping[str, tuple[str, ...]]:
    """Return the subcommand path of each per-command console script, e.g. plotartisestimators.

    The CommandSpec of a subcommand names its script, thus the tree holds both names. No second table
    can disagree with the tree.
    """
    scripts: dict[str, tuple[str, ...]] = {}

    def walk(tree: CommandTree, path: tuple[str, ...]) -> None:
        for name, spec in tree.items():
            if isinstance(spec, dict):
                walk(spec, (*path, name))
            elif spec.script:
                if spec.script in scripts:
                    msg = f"Console script {spec.script} names more than one subcommand"
                    raise ValueError(msg)
                scripts[spec.script] = (*path, name)

    walk(subcommandtree, ())

    return MappingProxyType(scripts)


def get_subcommand_of_script(scriptname: str) -> tuple[str, ...]:
    """Return the subcommand that a per-command console script stands for, or an empty tuple."""
    return get_script_subcommands().get(scriptname, ())


def addcommandargs(parser: argparse.ArgumentParser, spec: CommandSpec) -> None:
    """Add the arguments of one subcommand to a parser, and record how to run it."""
    submodule = importlib.import_module(f"artistools.{spec.module}")
    submodule.addargs(parser)
    # run_command alone implements --quiet, and __main__ implements --explain, thus no module declares them
    addarg_common(parser)
    if isinstance(parser, SuggestingArgumentParser):
        parser.add_standard_flags()
    # __main__ tests the arguments against the defaults of this parser, thus it needs the parser itself.
    # parse_cli_args cannot make that test, because it returns at once for a parsed namespace, which is
    # what the dispatcher gives it
    parser.set_defaults(func=getattr(submodule, spec.funcname), argparser=parser)


def build_script_parser(scriptname: str) -> argparse.ArgumentParser | None:
    """Return a parser for the subcommand of a per-command console script, or None for another name.

    A script such as plotartisestimators runs one subcommand, thus its parser holds that command alone.
    The usage text then gives the name of the script and not the name of the subcommand, and argparse
    builds one parser in place of the whole tree.
    """
    words = get_subcommand_of_script(scriptname)
    if not words:
        return None

    node: CommandSpec | CommandTree = subcommandtree
    for word in words:
        assert isinstance(node, dict)
        node = node[word]

    assert isinstance(node, CommandSpec)
    # the help of the script shows the same note and examples as the help of the subcommand
    parser = SuggestingArgumentParser(
        prog=scriptname,
        description=node.get_description(),
        epilog=get_command_epilog(words, node),
        formatter_class=CustomArgHelpFormatter,
    )
    addcommandargs(parser, node)

    return parser


def addsubparsers(parser: argparse.ArgumentParser, subcommandtree: CommandTree, path: tuple[str, ...] = ()) -> None:
    """Register the subcommands in the tree on the parser. path holds the names of the groups above the tree."""

    def func(args: argparse.Namespace) -> None:
        # the help is the product, thus it goes to the standard output and not to the standard error
        parser.print_help(getattr(args, "productstream", None))

    parser.set_defaults(func=func)
    subparsers = parser.add_subparsers(
        dest="subcommand", required=False, metavar="<command>", parser_class=SuggestingArgumentParser
    )
    # the subcommands are the only positional argument, and the headings of their groups replace this heading
    parser._positionals.title = argparse.SUPPRESS  # ruff:ignore[private-member-access]

    for subcommand, spec in subcommandtree.items():
        if isinstance(spec, dict):
            subparser = subparsers.add_parser(
                subcommand,
                help=GROUPHELPTEXT.get(subcommand, "Command group."),
                description=f"The {subcommand} commands of artistools.",
                epilog=get_group_epilog((*path, subcommand), spec),
                formatter_class=CommandListFormatter,
            )
            addsubparsers(parser=subparser, subcommandtree=spec, path=(*path, subcommand))
        else:
            # omitting help= entirely keeps a hidden entry out of the parent help listing. Do not use
            # help=argparse.SUPPRESS here: argparse only honours it for arguments, not subparsers, and
            # it would show the command with a literal ==SUPPRESS== description
            addparserkwargs: dict[str, t.Any] = {} if spec.hidden else {"help": spec.helptext}
            subparser = subparsers.add_parser(
                subcommand,
                description=spec.get_description(),
                epilog=get_command_epilog((*path, subcommand), spec),
                aliases=spec.aliases,
                formatter_class=CustomArgHelpFormatter,
                **addparserkwargs,
            )
            addcommandargs(subparser, spec)


def get_version_text() -> str:
    """Return the version of artistools. For a git checkout, also give the commit and its date, as ruff does.

    A bug report then names the code that ran, also between two releases, e.g. "artistools 2026.10.19 (7632bc6
    2026-10-20)".
    """
    import subprocess  # ruff:ignore[suspicious-subprocess-import]
    from importlib.metadata import version

    versiontext = f"artistools {version('artistools')}"
    repository = get_path("artistools_repository")
    # an installed wheel has no git folder, and a worktree has a .git file
    if not (repository / ".git").exists():
        return versiontext
    try:
        commit = subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
            ["git", "-C", str(repository), "log", "-1", "--format=%h %cs"],  # ruff:ignore[start-process-with-partial-path]
            capture_output=True,
            encoding="utf-8",
            check=True,
            timeout=5,
        ).stdout.strip()
    # a missing git or a folder that is not a repository gives the version alone
    except (OSError, subprocess.SubprocessError):
        return versiontext
    return f"{versiontext} ({commit})" if commit else versiontext


class VersionAction(argparse.Action):
    """Print the text of get_version_text and stop, as the version action of argparse does.

    The version action of argparse takes its text when the code makes the parser. Thus, with that action, git runs
    for each command.
    """

    def __init__(self, option_strings: "Sequence[str]", dest: str = argparse.SUPPRESS, **kwargs: t.Any) -> None:
        """Make an action that takes no value."""
        super().__init__(option_strings, dest=dest, nargs=0, default=argparse.SUPPRESS, **kwargs)

    @t.override
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: "str | Sequence[t.Any] | None",
        option_string: str | None = None,
    ) -> None:
        print(get_version_text())
        parser.exit()


def show_version(args: argparse.Namespace | None = None, **kwargs: t.Any) -> None:  # ruff:ignore[unused-function-argument]
    """Print the artistools version. The version is the product of the command, thus --quiet keeps it."""
    print_product(args or argparse.Namespace(), get_version_text())


def show_python(args: argparse.Namespace | None = None, **kwargs: t.Any) -> None:  # ruff:ignore[unused-function-argument]
    """Print the path of the Python interpreter that runs artistools. The path is the product, thus --quiet keeps it.

    The command of artistools can be in a different folder from its interpreter, e.g. ~/.local/bin for pip install
    --user, thus the folder of the command does not give the interpreter.
    """
    print_product(args or argparse.Namespace(), sys.executable)


def get_artistools_path(args: argparse.Namespace | None = None, **kwargs: t.Any) -> None:  # ruff:ignore[unused-function-argument]
    """Print the installed artistools package directory. The path is the product, thus --quiet keeps it."""
    print_product(args or argparse.Namespace(), get_path("artistools_dir"))


def get_older_flag_spellings(parser: argparse.ArgumentParser) -> list[str]:
    """Return the flags of the parser and of its subcommands that the help hides as an older spelling.

    argcomplete offers each option string of an action, thus --print<Tab> gave --print_data and --print-data.
    """
    older: set[str] = set()
    for action in parser._actions:  # ruff:ignore[private-member-access]
        if isinstance(action, argparse._SubParsersAction):  # ruff:ignore[private-member-access]
            for subparser in action.choices.values():
                older.update(get_older_flag_spellings(subparser))
        else:
            older.update(set(action.option_strings) - set(get_shown_flags(action)))
    return sorted(older)


def get_hidden_commands() -> list[str]:
    """Return the top-level commands that the help hides, thus tab completion offers none of them."""
    return [name for name, spec in subcommandtree.items() if isinstance(spec, CommandSpec) and spec.hidden]
