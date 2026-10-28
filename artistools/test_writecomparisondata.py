import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import artistools as at

modelpath = at.get_path("testdata") / "testmodel"
modelpath_classic_3d = at.get_path("testdata") / "test-classicmode_3d"
outputpath = at.get_path("testoutput")


@pytest.mark.benchmark
def test_writecomparisondata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The files take the name of the model folder, and the spectra keep the luminosity of spec.out.

    The default model path "." gave an empty model name, e.g. spectra__artisnebular.txt.
    """
    monkeypatch.chdir(modelpath)
    # CodSpeed runs a benchmark more than one time in one folder, thus the later runs find the files of the first
    at.writecomparisondata.main(
        argsraw=[], modelpath=Path(), outputpath=tmp_path, selected_timesteps=list(range(99)), force=True
    )

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "edep_testmodel_artisnebular.txt",
        "ionfrac_co_testmodel_artisnebular.txt",
        "ionfrac_fe_testmodel_artisnebular.txt",
        "phys_testmodel_artisnebular.txt",
        "spectra_testmodel_artisnebular.txt",
    ]

    from artistools.spectra.core import scan_spec

    # the spectra give L_lambda, and spec.out gives F_nu at 1 Mpc
    dfspecout = scan_spec(modelpath).collect()
    arr_specout_nu = dfspecout["nu"].to_numpy()
    arr_written = np.loadtxt(tmp_path / "spectra_testmodel_artisnebular.txt", comments="#")
    assert arr_written.shape == (len(arr_specout_nu), 1 + 99)
    area = 4.0 * math.pi * at.constants.megaparsec_to_cm**2
    for timestep in (10, 50):
        lum_written = np.trapezoid(arr_written[:, 1 + timestep], arr_written[:, 0])
        lum_specout = abs(np.trapezoid(dfspecout[:, 1 + timestep].to_numpy(), arr_specout_nu)) * area
        assert np.isclose(lum_written, lum_specout, rtol=1e-4)

    # the ion fractions of an element sum to one, as the format of the workshop requires
    arr_ionfrac_fe = np.loadtxt(tmp_path / "ionfrac_fe_testmodel_artisnebular.txt", comments="#")
    assert np.allclose(arr_ionfrac_fe[:, 1:].sum(axis=1), 1.0, rtol=1e-3)


def test_write_lbol_edep(tmp_path: Path) -> None:
    """The bolometric luminosity / deposition writer must produce one line per selected timestep.

    test_writecomparisondata() uses a model with no deposition.out, so its FileNotFoundError guard skipped this
    function entirely and hid the fact that it read column names the light curve frame does not have.
    """
    from artistools.writecomparisondata import write_lbol_edep

    outputpath = tmp_path / "lbol_edep.txt"
    selected_timesteps = [0, 1, 2, 5]
    write_lbol_edep(modelpath_classic_3d, selected_timesteps, outputpath)

    lines = outputpath.read_text(encoding="utf-8").splitlines()
    assert lines[0] == f"#NTIMES: {len(selected_timesteps)}"
    datalines = [line for line in lines if not line.startswith("#")]
    assert len(datalines) == len(selected_timesteps)
    for line in datalines:
        timedays, lbol, edep = (float(x) for x in line.split())
        assert timedays > 0.0
        assert lbol > 0.0
        assert edep > 0.0


def test_write_lbol_edep_ntimes_matches_rows(tmp_path: Path) -> None:
    """The NTIMES header must count the rows actually written, not the timesteps asked for.

    A selected timestep with no light curve or deposition data is dropped by the join, so a header quoting
    len(selected_timesteps) would promise more times than the file contains and misalign every reader.
    """
    from artistools.writecomparisondata import write_lbol_edep

    outputpath = tmp_path / "lbol_edep.txt"
    write_lbol_edep(modelpath_classic_3d, [0, 1, 2, 5, 9999], outputpath)

    lines = outputpath.read_text(encoding="utf-8").splitlines()
    datalines = [line for line in lines if not line.startswith("#")]
    assert lines[0] == f"#NTIMES: {len(datalines)}"
    # timestep 9999 does not exist, so it must not be counted
    assert len(datalines) == 4


def test_writecomparisondata_rejects_an_empty_timestep_list(tmp_path: Path) -> None:
    """An empty timestep list wrote files that hold a header and no data."""
    with pytest.raises(ValueError, match="selected-timesteps"):
        at.writecomparisondata.main(argsraw=[], modelpath=modelpath, outputpath=tmp_path, selected_timesteps=[])


def test_writecomparisondata_of_a_model_with_no_estimators_gives_a_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A model with no estimator files stopped with a raw ColumnNotFoundError for Te."""
    modelfolder = tmp_path / "noestimators"
    modelfolder.mkdir()
    for sourcefile in modelpath.iterdir():
        if sourcefile.is_file() and not sourcefile.name.startswith("estimators"):
            (modelfolder / sourcefile.name).symlink_to(sourcefile.resolve())

    with pytest.raises(SystemExit) as excinfo:
        at.writecomparisondata.main(
            argsraw=[], modelpath=modelfolder, outputpath=tmp_path / "out", selected_timesteps=[50]
        )

    assert excinfo.value.code == 1
    assert "holds no estimator files" in capsys.readouterr().err


def test_ionfrac_header_counts_the_stages_from_neutral(tmp_path: Path) -> None:
    """The format labels the neutral stage 0, thus the stages below the lowest ion of an element hold zero.

    The header gave each column its ARTIS ion stage, which is 1 for the neutral stage, thus Co II took the label co2.
    """
    at.writecomparisondata.main(
        argsraw=[], modelpath=modelpath, outputpath=tmp_path, selected_timesteps=list(range(10))
    )

    ionfracfiles = sorted(tmp_path.glob("ionfrac_*_artisnebular.txt"))
    assert ionfracfiles, "the run wrote no ion fraction file"

    elementlist = at.atomic.get_composition_data(modelpath)
    lowermost_of_elsymbol = {
        at.get_elsymbol(row["Z"]).lower(): row["lowermost_ion_stage"] for row in elementlist.iter_rows(named=True)
    }
    assert any(lowermost > 1 for lowermost in lowermost_of_elsymbol.values())

    for ionfracfile in ionfracfiles:
        elsymbol = ionfracfile.name.split("_")[1]
        lines = ionfracfile.read_text(encoding="utf-8").splitlines()
        nstages = int(next(line for line in lines if line.startswith("#NSTAGES:")).split()[1])
        headers = [line for line in lines if line.startswith("#vel_mid")]
        assert headers
        for header in headers:
            assert header.split()[1:] == [f"{elsymbol}{stage}" for stage in range(nstages)]

        datarows = [line.split()[1:] for line in lines if not line.startswith("#")]
        assert datarows
        for row in datarows:
            assert len(row) == nstages
            assert all(float(value) == 0.0 for value in row[: lowermost_of_elsymbol[elsymbol] - 1])


def test_writecomparisondata_keeps_a_tiny_ion_fraction(tmp_path: Path) -> None:
    """An ion fraction below 1e-38 must keep its digits.

    The estimator cache stores a population as Float32. A Float32 ratio below 1e-38 is subnormal, thus it lost
    digits or became zero. The old reader divided Python floats, and the files of the two readers differed.
    """
    from artistools.writecomparisondata import write_ionfracts

    dfestimators = pl.DataFrame(
        {"timestep": [0], "vel_r_mid": [1e9], "nnelement_Fe": [1e8], "nnion_Fe_II": [1e-37]},
        schema={"timestep": pl.Int32, "vel_r_mid": pl.Float64, "nnelement_Fe": pl.Float32, "nnion_Fe_II": pl.Float32},
    )
    write_ionfracts(modelpath, "tiny", [0], dfestimators, tmp_path)

    datalines = [
        line
        for line in (tmp_path / "ionfrac_fe_tiny_artisnebular.txt").read_text().splitlines()
        if not line.startswith("#")
    ]
    expected = float(np.float32(1e-37)) / float(np.float32(1e8))
    # the columns are the velocity, then the stages from the neutral stage, thus Fe II is the third value
    assert float(datalines[0].split()[2]) == pytest.approx(expected, rel=1e-4, abs=0.0)


def test_writecomparisondata_edep_takes_the_next_timestep(tmp_path: Path) -> None:
    """ARTIS writes the deposition of timestep n into the estimators of timestep n + 1.

    The edep file took the row of timestep n, thus it gave the deposition of the timestep before, and it
    disagreed with the lbol_edep file of the same export by about 5 % at 200 days.
    """
    at.writecomparisondata.main(argsraw=[], modelpath=modelpath, outputpath=tmp_path, selected_timesteps=[50])

    datalines = [
        line for line in (tmp_path / "edep_testmodel_artisnebular.txt").read_text().splitlines() if line[0] != "#"
    ]
    nextrow = at.scan_estimators(modelpath=modelpath, timestep=(51,)).select("total_dep").collect()
    assert float(datalines[0].split()[1]) == pytest.approx(nextrow.item(), rel=1e-4, abs=0.0)

    # ARTIS wrote no timestep after the last one, thus the file gives zero there and the export goes on
    lasttimestep = len(at.get_timestep_times(modelpath)) - 1
    at.writecomparisondata.main(
        argsraw=[], modelpath=modelpath, outputpath=tmp_path, selected_timesteps=[lasttimestep], force=True
    )
    datalines = [
        line for line in (tmp_path / "edep_testmodel_artisnebular.txt").read_text().splitlines() if line[0] != "#"
    ]
    assert float(datalines[0].split()[1]) == 0.0
