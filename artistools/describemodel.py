"""Summarise an ARTIS model: its folders, cells, timesteps, output files, ions, ARTIS version, and physics options.

The summary reads the model file, the names of the output files, the start of the first log of each run, and
artisoptions.h. It reads no other large file. Run it first to choose the readers and the timesteps for a question about
a model, and to know which physics the run had on.
"""

import argparse
import math
import operator
import re
import typing as t
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType

import polars as pl

from artistools.atomic import get_composition_data
from artistools.atomic import get_ionstring
from artistools.inputmodel import add_derived_cols_to_modeldata
from artistools.inputmodel import get_modeldata
from artistools.lightcurve import scan_lightcurve
from artistools.logsummary import get_logfiles
from artistools.logsummary import RE_TIMESTEP
from artistools.misc import addarg_modelpath
from artistools.misc import firstexisting_or_none
from artistools.misc import get_artis_options
from artistools.misc import get_model_folder
from artistools.misc import get_rank_outputfiles
from artistools.misc import get_runfolders
from artistools.misc import get_timestep_times
from artistools.misc import get_viewingdirection_costhetabincount
from artistools.misc import get_viewingdirection_phibincount
from artistools.misc import get_viewingdirectionbincount
from artistools.misc import get_vpkt_config
from artistools.misc import on_model_host
from artistools.misc import parse_cli_args
from artistools.misc import print_product
from artistools.misc import zopen
from artistools.packets import count_packets_textfiles
from artistools.packets import has_packets_files

# the output files that are not rank files, in the order of the summary. A reader also accepts each one with a
# compressed extension
SINGLEOUTPUTFILES = (
    "light_curve.out",
    "light_curve_res.out",
    "gamma_light_curve.out",
    "spec.out",
    "spec_res.out",
    "gamma_spec.out",
    "deposition.out",
)

# the artistools command that reads each output, as a hint for a person or an agent. <days> is a time in days, N is a
# direction bin, and <cell> is a cell
READERCOMMANDS: Mapping[str, str] = MappingProxyType({
    "estimators": "plotestimators <variable> -t <days>; deposition; check",
    "nlte": "plotnltepops -t <days> --modelgridindex <cell>",
    "packets": "plotspectra --frompackets; plotlightcurves --frompackets; plotlastpacketinteraction; plotspherical",
    "light_curve.out": "plotlightcurves",
    "light_curve_res.out": "plotlightcurves --plotviewingangle N",
    "gamma_light_curve.out": "plotlightcurves --gamma",
    "spec.out": "plotspectra -t <days>",
    "spec_res.out": "plotspectra -t <days> --plotviewingangle N",
    "gamma_spec.out": "plotspectra --gamma -t <days>",
    "deposition.out": "plotlightcurves --deposition gamma positron",
})


def get_reader_text(outputname: str) -> str:
    """Return the text that names the artistools command that reads an output."""
    return f" (read with: artistools {READERCOMMANDS[outputname]})" if outputname in READERCOMMANDS else ""


# the options of artisoptions.h that change the physics of a run or the meaning of an output, by topic. ARTIS renamed
# some options, thus each entry gives the names of one option, the newest name first
PHYSICSOPTIONS: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = (
    (
        "NLTE",
        (
            ("ION_NLEVELS_EXCITED_NLTE", "LEVEL_IS_NLTE"),
            ("FORCE_SAHA_ION_BALANCE",),
            ("NLTE_TIME_DEPENDENT_FIRST_TIMESTEP",),
            ("NLTE_TE_NNE_MAXITER", "NLTEITER"),
        ),
    ),
    ("non-thermal", (("NT_SCHEME", "NT_ON"), ("NT_SOLVE_SPENCERFANO",), ("NT_EXCITATION_ON",))),
    (
        "thermalisation",
        (("PARTICLE_THERMALISATION_SCHEME", "INSTANT_PARTICLE_DEPOSITION"), ("GAMMA_THERMALISATION_SCHEME",)),
    ),
    (
        "radiation field",
        (
            ("MULTIBIN_RADFIELD_MODEL_ON",),
            ("MULTIBIN_RADFIELD_FIRST_TIMESTEP", "FIRST_BINNED_RADFIELD_TIMESTEP", "FIRST_NLTE_RADFIELD_TIMESTEP"),
            ("LTEPOP_EXCITATION_USE_TJ",),
        ),
    ),
    (
        "estimators",
        (
            ("DETAILED_BF_ESTIMATORS_ON",),
            ("DETAILED_LINE_ESTIMATORS_ON",),
            ("USE_LUT_PHOTOION",),
            ("WRITE_ION_HEATING_COOLING_RATES",),
        ),
    ),
    ("temperature limits", (("MINTEMP",), ("MAXTEMP",))),
    (
        "packets",
        (("NUM_PACKETS", "MPKTS"), ("POL_ON",), ("VPKT_ON",), ("KEEP_ESCAPED_GAMMAS",), ("SAMPLE_RPKT_EMISSION",)),
    ),
)


def get_model_lines(modelpath: Path) -> list[str]:
    """Return the lines that describe the cells of model.txt."""
    dfmodel, modelmeta = get_modeldata(modelpath, printwarningsonly=True)
    cellswithmatter = (
        add_derived_cols_to_modeldata(dfmodel, modelmeta=modelmeta).select((pl.col("rho") > 0).sum()).collect().item()
    )
    return [
        f"model.txt: {modelmeta['dimensions']}D, {modelmeta['npts_model']} cells, {cellswithmatter} cells with matter",
        f"  t_model_init_days {modelmeta['t_model_init_days']:g}, vmax {modelmeta['vmax_cmps']:.4g} cm/s",
    ]


def get_timestep_lines(modelpath: Path) -> list[str]:
    """Return the line that gives the number of timesteps and the days that they cover."""
    tstarts = get_timestep_times(modelpath, loc="start")
    tends = get_timestep_times(modelpath, loc="end")
    return [
        f"timesteps: {len(tstarts)}, from {tstarts[0]:.3f} to {tends[-1]:.3f} days (`artistools timesteps` lists them)"
    ]


def get_output_lines(modelpath: Path) -> list[str]:
    """Return the lines that name the output files of the model and of its run folders."""
    runfolders = get_runfolders(modelpath)
    runfoldernames = [str(folder.relative_to(modelpath)) if folder != modelpath else "." for folder in runfolders]
    lines = [f"run folders with estimators: {', '.join(runfoldernames) or 'none'}"]
    for kind, filenameformat in (("estimators", "estimators_{mpirank:04d}.out"), ("nlte", "nlte_{mpirank:04d}.out")):
        rankcount = sum(len(get_rank_outputfiles(folder, filenameformat)) for folder in runfolders)
        # a run can join the files of its ranks into one file, which the readers of the rank files also read
        allrankscount = sum(
            firstexisting_or_none(f"{kind}_allranks.out", folder=folder, search_subfolders=False) is not None
            for folder in runfolders
        )
        alltext = f", {allrankscount} files of all ranks ({kind}_allranks.out)" if allrankscount else ""
        lines.append(f"{kind}: {rankcount} rank files{alltext}{get_reader_text(kind) if rankcount or alltext else ''}")

    packetcount, otherrankcount = count_packets_textfiles(modelpath)
    if packetcount == 0 and has_packets_files(modelpath):
        lines.append("packets: no text files, and the parquet cache of the packets")
    else:
        lines.append(f"packets: {packetcount} rank files{get_reader_text('packets') if packetcount else ''}")
    if otherrankcount:
        # a restart with fewer ranks leaves the files of the higher ranks of an earlier run
        lines.append(
            f"  {otherrankcount} more packets files have a rank above the ranks of input.txt, thus the readers do not"
            " read them. They can come from an earlier run with more ranks"
        )

    for filename in SINGLEOUTPUTFILES:
        # the readers of the light curves and the spectra also search the run folders, and get_deposition does not
        filepath = firstexisting_or_none(filename, folder=modelpath, search_subfolders=filename != "deposition.out")
        lines.append(
            f"{filename}: {filepath.relative_to(modelpath)}{get_reader_text(filename)}"
            if filepath is not None
            else f"{filename}: none"
        )
    return lines


def get_selection_lines(modelpath: Path) -> list[str]:
    """Return the lines that name what a plot can select: the direction bins, the virtual packet observers, and more.

    An agent then gives a valid selection with no trial of the arguments.
    """
    lines = ["plot selections:"]
    lcresfile = firstexisting_or_none("light_curve_res.out", folder=modelpath)
    if lcresfile is not None:
        # ARTIS sets MABINS when it compiles, thus the file and not artistools gives the count of the bins
        bincount = len(scan_lightcurve(lcresfile, directionresolved=True))
        geometrytext = (
            f" = {get_viewingdirection_costhetabincount()} cos θ bins times {get_viewingdirection_phibincount()} φ bins"
            if bincount == get_viewingdirectionbincount()
            else f", and artistools gives the angles of {get_viewingdirectionbincount()} bins"
        )
        lines.append(
            f"  direction bins: {bincount} tables in {lcresfile.relative_to(modelpath)}{geometrytext}."
            f" --plotviewingangle N selects bin N of 0 to {bincount - 1}, and -1 is all the directions"
        )
    elif firstexisting_or_none("spec_res.out", folder=modelpath) is not None:
        lines.append(
            f"  direction bins: spec_res.out, with {get_viewingdirectionbincount()} bins if ARTIS has the default"
            " MABINS. --plotviewingangle N selects bin N, and -1 is all the directions"
        )
    if Path(modelpath, "vpkt.txt").is_file():
        vpktconfig = get_vpkt_config(modelpath)
        timetext = (
            f", from {vpktconfig['initial_time']} to {vpktconfig['final_time']} d"
            if vpktconfig["time_limits_enabled"]
            else ""
        )
        spectrumcount = vpktconfig["nobsdirections"] * vpktconfig["nspectraperobs"]
        lines.append(
            f"  virtual packet spectra: {spectrumcount} = {vpktconfig['nobsdirections']} directions times"
            f" {vpktconfig['nspectraperobs']} opacity choices{timetext}. --plotvspecpol N selects spectrum N of 0 to"
            f" {spectrumcount - 1}"
        )
    if get_runfolders(modelpath):
        lines.append(
            "  estimator variables: plotestimators --listvariables lists them, and --listnuclides lists the nuclides"
        )
    return lines if len(lines) > 1 else []


def get_ion_lines(modelpath: Path) -> list[str]:
    """Return one line for each element of compositiondata.txt, with its lowest and its top ion."""
    dfcomposition = get_composition_data(modelpath)
    lines = ["ions (ARTIS cannot ionise an element above its top ion):"]
    lines.extend(
        f"  {get_ionstring(z, lowest)} to {get_ionstring(z, top)}"
        for z, lowest, top in dfcomposition.select("Z", "lowermost_ion_stage", "uppermost_ion_stage").iter_rows()
    )
    return lines


def get_option_lines(modelpath: Path, alloptions: bool = False) -> list[str]:
    """Return the physics options of artisoptions.h by topic, or each option of the file with alloptions."""
    options = get_artis_options(modelpath)
    if not options:
        return ["artisoptions.h: none in the model folder or in its folder artis"]
    if alloptions:
        return ["artisoptions.h:", *(f"  {name} = {value}" for name, value in options.items())]

    lines = ["artisoptions.h (--alloptions gives each option):"]
    for topic, optionnames in PHYSICSOPTIONS:
        found = [
            f"{name} = {options[name]}"
            for names in optionnames
            if (name := next((name for name in names if name in options), None)) is not None
        ]
        if found:
            lines.append(f"  {topic}: {'; '.join(found)}")
            lines.extend(f"    effect: {effect}" for effect in get_option_effects(topic, options))
    return lines


# the method of each time-dependent scheme, from the comments of update_packets.cc of ARTIS
THERMALISATIONMETHODS: Mapping[str, str] = MappingProxyType({
    "BARNES": (
        "a particle packet deposits with the efficiency of Barnes et al. (2016), ApJ, 829, 110,"
        " doi:10.3847/0004-637X/829/2/110, and otherwise escapes"
    ),
    "WOLLAEGER": (
        "a particle packet deposits with the efficiency of Wollaeger et al. (2018), MNRAS, 478, 3298-3334,"
        " doi:10.1093/mnras/sty1018, and otherwise escapes"
    ),
    "TIMEDEPENDENT": (
        "the particles lose energy along their paths, as Shingles et al. (2023), ApJL, 954, L41,"
        " doi:10.3847/2041-8213/acf29a, describe"
    ),
    "TIMEDEPENDENT_WITH_ADIABATIC_LOSS": (
        "the particles lose energy along their paths and by the adiabatic expansion, as Shingles et al. (2023),"
        " ApJL, 954, L41, doi:10.3847/2041-8213/acf29a, describe"
    ),
})


def returns_only(value: str | None, result: str) -> bool:
    """Return True if the text of a function option has only return statements of one result, e.g. "false"."""
    if value is None:
        return False
    results = re.findall(r"\breturn\s+([^;]+);", value)
    return bool(results) and all(found.strip() == result for found in results)


def get_option_effects(topic: str, options: Mapping[str, str]) -> list[str]:
    """Return what the options of a topic mean for the outputs, from the physics notes of the artis-data skill.

    A value that the code cannot read, e.g. a function with a condition, gives no effect line. The option line above
    it still gives the value.
    """
    effects: list[str] = []
    match topic:
        case "NLTE":
            nlteoption = options.get("ION_NLEVELS_EXCITED_NLTE", options.get("LEVEL_IS_NLTE"))
            hasnonltelevel = returns_only(nlteoption, "false") or returns_only(nlteoption, "0")
            sahaoption = options.get("FORCE_SAHA_ION_BALANCE")
            # phi_saha of ltepop.cc of ARTIS evaluates the Saha equation at Te
            if returns_only(sahaoption, "true"):
                effects.append("the ion balance of each element is the Saha equation at Te")
            elif hasnonltelevel:
                effects.append("the ratio of two ions comes from the balance of the ionisation and the recombination")
            if (
                sahaoption is not None
                and not returns_only(sahaoption, "true")
                and not returns_only(sahaoption, "false")
            ):
                effects.append(
                    "FORCE_SAHA_ION_BALANCE gives the Saha equation at Te to the ion balance of some elements"
                )
            if hasnonltelevel:
                effects.append("no level is NLTE, thus the levels of an ion follow the Boltzmann distribution")
            elif nlteoption is not None:
                effects.append(
                    "ARTIS solves the rate equations of the NLTE levels and of a superlevel of each ion, as section 2.3"
                    " of Shingles et al. (2020), MNRAS, 492, 2029-2043, doi:10.1093/mnras/stz3412, describes"
                )
        case "non-thermal":
            ntscheme = options.get("NT_SCHEME", "")
            if options.get("NT_ON") == "false" or "NT_OFF" in ntscheme:
                effects.append("ARTIS gives no non-thermal ionisation and no non-thermal excitation")
            elif "NT_AXELRODAPPROX" in ntscheme:
                effects.append("the non-thermal rates come from the approximation NT_AXELRODAPPROX, not Spencer-Fano")
            elif "NT_SPENCERFANO" in ntscheme or options.get("NT_SOLVE_SPENCERFANO") == "true":
                effects.append(
                    "the Spencer-Fano solution gives the fractions of the deposition that go into heating, ionisation,"
                    " and excitation, and the rates gamma_NT_<ion>"
                )
        case "thermalisation":
            scheme = options.get("PARTICLE_THERMALISATION_SCHEME")
            instant = "INSTANTFULLDEPOSITION" in scheme if scheme is not None else None
            if scheme is None and "INSTANT_PARTICLE_DEPOSITION" in options:
                instant = options["INSTANT_PARTICLE_DEPOSITION"] == "true"
            if instant is True:
                effects.append(
                    "the positrons, the electrons, and the alpha particles give their energy at once to their cell,"
                    " thus their deposition_ rates are analytic and belong to the time step of their row"
                )
            elif instant is False:
                effects.append(
                    "the particles are Monte Carlo packets, thus their deposition_ rates belong to the time step"
                    " before their row"
                )
                for schemename, method in THERMALISATIONMETHODS.items():
                    if scheme is not None and scheme.endswith(f"::{schemename}"):
                        effects.append(method)
        case "radiation field":
            if options.get("MULTIBIN_RADFIELD_MODEL_ON") == "true":
                effects.append(
                    "from the first time step of the binned radiation field (the option above), the photoionisation"
                    " rates use a fit of TR and W to each frequency bin of J"
                )
            if options.get("LTEPOP_EXCITATION_USE_TJ") == "true":
                effects.append("the Boltzmann distribution of the levels uses TJ in place of Te")
        case "temperature limits":
            effects.append("a cell with Te at MINTEMP or MAXTEMP has no thermal balance, thus that Te is not physical")
        case "packets":
            if options.get("VPKT_ON") == "true":
                effects.append("the virtual packets give a spectrum for each direction of vpkt.txt, with less noise")
            if options.get("POL_ON") == "true":
                effects.append("specpol.out and specpol_res.out give the Stokes parameters of the spectra")
        case _:
            # the estimators topic names the columns that ARTIS writes, and it changes no physics
            pass
    return effects


def get_run_versions(modelpath: Path) -> dict[str, dict[str, str]]:
    """Return the git version and the git branch of ARTIS of each run folder with a log, in the order of the runs.

    A restarted run can use a different build of ARTIS, thus each run folder gives its own version. The key is the
    name of the run folder, or "." for the model folder. A log can give one of the two fields, or none. A copy of a log,
    e.g. output_0-0.txt of the last run in the model folder, is not a run, thus it gives no version. A restart starts
    at a later timestep, thus the first timestep of each log gives the order. A log with no timestep keeps its place
    after the log before it in the order of the folders.
    """
    runs: list[tuple[float, str, dict[str, str]]] = []
    for logpath in get_logfiles(modelpath, [0]):
        fields: dict[str, str] = {}
        firsttimestep: int | None = None
        with zopen(logpath) as flog:
            for linenumber, line in enumerate(flog):
                # ARTIS writes the git lines near the start of the log, thus the search for them reads the first lines
                if linenumber < 500:
                    for key in ("git version", "git branch"):
                        if key in line and key not in fields:
                            fields[key] = line.split(key, 1)[1].lstrip(": ").strip()
                if (timestepmatch := RE_TIMESTEP.search(line)) is not None:
                    firsttimestep = int(timestepmatch.group(1))
                    break
        orderkey = float(firsttimestep) if firsttimestep is not None else (runs[-1][0] if runs else -math.inf)
        # a log with no git line is still a run, thus the last run is not the run before it, and its fields are empty
        runs.append((orderkey, "." if logpath.parent == modelpath else logpath.parent.name, fields))
    return {foldername: fields for _, foldername, fields in sorted(runs, key=operator.itemgetter(0))}


def get_version_lines(modelpath: Path) -> list[str]:
    """Return the ARTIS version and the git branch of each run, from the start of its first log."""
    return [
        f"ARTIS of run folder {foldername}: {fields.get('git version', 'unknown version')}"
        f" (branch {fields.get('git branch', 'unknown')})"
        for foldername, fields in get_run_versions(modelpath).items()
    ]


@on_model_host
def get_model_summary(modelpath: Path, alloptions: bool = False) -> list[str]:
    """Return the lines of the summary. A part whose input file is missing gives one line that says so.

    The host of a remote model makes the summary, because each part reads a file of the model folder.
    """
    lines = [f"model folder: {modelpath.resolve()}"]
    parts = (
        get_model_lines,
        get_timestep_lines,
        get_output_lines,
        get_selection_lines,
        get_ion_lines,
        get_version_lines,
    )
    for part in parts:
        try:
            lines.extend(part(modelpath))
        except FileNotFoundError as exc:
            lines.append(f"{part.__name__.removeprefix('get_').removesuffix('_lines')}: {exc}")
    lines.extend(get_option_lines(modelpath, alloptions=alloptions))
    return lines


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, positional=True, default=Path())
    parser.add_argument(
        "--alloptions", action="store_true", help="Give each option of artisoptions.h, and not only the physics options"
    )


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Summarise an ARTIS model: its folders, cells, timesteps, output files, ions, ARTIS version, and options."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)
    modelpath = get_model_folder(args.modelpath)
    print_product(args, "\n".join(get_model_summary(modelpath, alloptions=args.alloptions)))
