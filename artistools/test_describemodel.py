import shutil
from pathlib import Path

import pytest

import artistools as at

modelpath = at.get_path("testdata") / "testmodel"
modelpath_classic_3d = at.get_path("testdata") / "test-classicmode_3d"


def test_describe_counts_the_cells_and_the_rank_files_of_each_run_folder(capsys: pytest.CaptureFixture[str]) -> None:
    """The summary must count the cells with matter and the rank files of all the run folders, e.g. job0 and job1."""
    import artistools.__main__

    artistools.__main__.main(argsraw=["describe", str(modelpath_classic_3d), "--quiet"])
    output = capsys.readouterr().out
    assert "1000 cells, 695 cells with matter" in output
    assert "timesteps: 30, from 3.000 to 8.000 days" in output
    assert "run folders with estimators: job0, job1" in output
    assert "estimators: 4 rank files" in output
    assert "nlte: 0 rank files" in output
    assert "Fe I to Fe V" in output
    assert "direction bins: 100 tables in light_curve_res.out = 10 cos θ bins times 10 φ bins" in output
    # the model folder of the classic-mode 3D test holds artisoptions.h and no folder artis
    assert "temperature limits: MINTEMP = 3500.; MAXTEMP = 140000." in output
    # an old ARTIS gave a different name to an option, e.g. NT_ON and not NT_SCHEME
    assert "non-thermal: NT_ON = false" in output
    assert "ARTIS of run folder job0: v2023.10-106-gfcb7e2c (branch develop)" in output


def test_describe_gives_the_effect_of_the_physics_options() -> None:
    """The ion balance of a kilonova model with FORCE_SAHA_ION_BALANCE is Saha at Te, and not the rate balance.

    The effect of the options must follow the code of ARTIS, thus each case here comes from ltepop.cc and
    update_packets.cc.
    """
    from artistools.describemodel import get_option_effects

    kilonova = {
        "ION_NLEVELS_EXCITED_NLTE": "(int element_z, int ionstage) { return 0; }",
        "FORCE_SAHA_ION_BALANCE": "(int element_z) { return true; }",
        "NT_SCHEME": "NonThermalScheme::NT_OFF",
        "PARTICLE_THERMALISATION_SCHEME": "ParticleThermalisationScheme::BARNES",
    }
    assert get_option_effects("NLTE", kilonova) == [
        "the ion balance of each element is the Saha equation at Te",
        "no level is NLTE, thus the levels of an ion follow the Boltzmann distribution",
    ]
    assert get_option_effects("non-thermal", kilonova) == [
        "ARTIS gives no non-thermal ionisation and no non-thermal excitation"
    ]
    thermalisation = get_option_effects("thermalisation", kilonova)
    assert "belong to the time step before their row" in thermalisation[0]
    assert "Barnes et al. (2016)" in thermalisation[1]

    # a condition in the function of an option gives no claim about its effect
    nebular = {
        "ION_NLEVELS_EXCITED_NLTE": "(int element_z, int ionstage) { if (element_z == 26) { return 80; } return 0; }",
        "FORCE_SAHA_ION_BALANCE": "(int element_z) { if (element_z == 1) { return true; } return false; }",
        "INSTANT_PARTICLE_DEPOSITION": "true",
    }
    nebulareffects = get_option_effects("NLTE", nebular)
    assert "the Saha equation at Te to the ion balance of some elements" in nebulareffects[0]
    assert "rate equations of the NLTE levels" in nebulareffects[1]
    assert "analytic" in get_option_effects("thermalisation", nebular)[0]


def test_describe_names_the_packets_files_of_an_earlier_run_with_more_ranks(tmp_path: Path) -> None:
    """The count took the ranks of input.txt alone, thus it said nothing of the packets of an earlier run with more."""
    from artistools.describemodel import get_output_lines

    shutil.copy(modelpath_classic_3d / "input.txt", tmp_path / "input.txt")
    nprocs = at.misc.get_nprocs(tmp_path)
    (tmp_path / "100.slurm").mkdir()
    for rank in [*range(nprocs), nprocs + 1]:
        (tmp_path / "100.slurm" / f"packets00_{rank:04d}.out").write_text("", encoding="utf-8")

    lines = get_output_lines(tmp_path)
    assert any(line.startswith(f"packets: {nprocs} rank files") for line in lines)
    assert "  1 more packets files have a rank above the ranks of input.txt" in "\n".join(lines)

    # the search stopped at the model folder when it held every rank, thus the count missed the run folder
    for rank in range(nprocs):
        (tmp_path / "100.slurm" / f"packets00_{rank:04d}.out").rename(tmp_path / f"packets00_{rank:04d}.out")
    lines = get_output_lines(tmp_path)
    assert "  1 more packets files have a rank above the ranks of input.txt" in "\n".join(lines)

    # a second earlier run with the same rank gave one rank number, and the count gave one file for the two
    (tmp_path / "200.slurm").mkdir()
    (tmp_path / "200.slurm" / f"packets00_{nprocs + 1:04d}.out").write_text("", encoding="utf-8")
    assert "  2 more packets files have a rank above the ranks of input.txt" in "\n".join(get_output_lines(tmp_path))


def test_run_versions_follow_the_runs_and_leave_out_a_copied_log(tmp_path: Path) -> None:
    """A copy of the log of an earlier run in the model folder came last, thus the catalog gave its old version."""
    from artistools.catalog import get_version_columns
    from artistools.describemodel import get_run_versions

    timestepline = "timestep {}: time before update grid 0.1 (tstart + tdelta/2) 2.0 days\n"
    for foldername, version, firsttimestep in [("200.slurm", "newbuild", 10), ("100.slurm", "oldbuild", 0)]:
        (tmp_path / foldername).mkdir()
        (tmp_path / foldername / "output_0-0.txt").write_text(
            f"git version: {version}\n{timestepline.format(firsttimestep)}", encoding="utf-8"
        )
    shutil.copy(tmp_path / "100.slurm" / "output_0-0.txt", tmp_path / "output_0-0.txt")

    assert list(get_run_versions(tmp_path)) == ["100.slurm", "200.slurm"]
    assert get_version_columns(tmp_path) == {"artis_version": "newbuild"}

    # the log of the last run gives no version, and the catalog gave the version of the run before it
    (tmp_path / "300.slurm").mkdir()
    (tmp_path / "300.slurm" / "output_0-0.txt").write_text(timestepline.format(20), encoding="utf-8")
    assert list(get_run_versions(tmp_path)) == ["100.slurm", "200.slurm", "300.slurm"]
    assert get_version_columns(tmp_path) == {"artis_version": None}


def test_describe_counts_the_direction_bins_of_the_file(tmp_path: Path) -> None:
    """ARTIS sets MABINS when it compiles, and the summary gave the 100 bins of artistools for a file of 2 bins."""
    from artistools.describemodel import get_selection_lines

    (tmp_path / "light_curve_res.out").write_text("1.0 1.0 1.0\n2.0 1.0 1.0\n" * 2, encoding="utf-8")
    (selectionline,) = [line for line in get_selection_lines(tmp_path) if "direction bins" in line]
    assert "direction bins: 2 tables in light_curve_res.out, and artistools gives the angles of 100 bins" in (
        selectionline
    )
    assert "selects bin N of 0 to 1," in selectionline


def test_describe_finds_the_light_curve_in_a_run_folder(tmp_path: Path) -> None:
    """A run on a cluster writes light_curve.out into its run folder, and the summary searched only the model folder.

    The light curve reader searches the run folders, and scan_deposition reads only the model folder.
    """
    from artistools.describemodel import get_output_lines

    modelfolder = tmp_path / "model"
    (modelfolder / "12411848.slurm").mkdir(parents=True)
    shutil.copy(modelpath / "input.txt", modelfolder / "input.txt")
    (modelfolder / "12411848.slurm" / "light_curve.out").write_text("1.0 2.0 3.0\n", encoding="utf-8")
    (modelfolder / "12411848.slurm" / "deposition.out").write_text("1.0 2.0 3.0\n", encoding="utf-8")

    lines = get_output_lines(modelfolder)
    assert "light_curve.out: 12411848.slurm/light_curve.out (read with: artistools plotlightcurves)" in lines
    assert "deposition.out: none" in lines

    # a run can join the estimator files of its ranks into one file, and the summary counted the rank files alone
    allranksmodel = tmp_path / "allranksmodel"
    (allranksmodel / "12411848.slurm").mkdir(parents=True)
    shutil.copy(modelpath / "input.txt", allranksmodel / "input.txt")
    (allranksmodel / "12411848.slurm" / "estimators_allranks.out").write_text(
        "timestep 0 modelgridindex 0 TR 2000 Te 3000 W 1 TJ 2000 nne 1.0e5\n\n", encoding="utf-8"
    )
    lines = get_output_lines(allranksmodel)
    assert any(
        line.startswith("estimators: 0 rank files, 1 files of all ranks (estimators_allranks.out) (read with:")
        for line in lines
    )
