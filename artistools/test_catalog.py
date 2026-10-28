import io
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import artistools as at

modelpath = at.get_path("testdata") / "testmodel"
modelpath_classic_3d = at.get_path("testdata") / "test-classicmode_3d"


def read_data_file(path: Path) -> pl.DataFrame:
    """Read a table that --write_data wrote."""
    return pl.read_csv(path, separator=" ", null_values="nan", comment_prefix="#")


def test_catalog_gives_one_row_for_each_model_and_warns_about_a_broken_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A folder of models gives one row for each model, and a broken model gives null values and one warning."""
    from artistools.catalog import main

    (tmp_path / "group").mkdir()
    (tmp_path / "group" / "classic3d").symlink_to(modelpath_classic_3d, target_is_directory=True)
    (tmp_path / "testmodel").symlink_to(modelpath, target_is_directory=True)
    # a link back to the top folder must not make the search loop
    (tmp_path / "group" / "loop").symlink_to(tmp_path, target_is_directory=True)
    brokenfolder = tmp_path / "broken"
    brokenfolder.mkdir()
    (brokenfolder / "input.txt").write_text("not an input file\n", encoding="utf-8")
    (brokenfolder / "model.txt").write_text("not a model\n", encoding="utf-8")

    outputfile = tmp_path / "out" / "catalog.txt"
    main(argsraw=[str(tmp_path), "-o", str(outputfile)])
    captured = capsys.readouterr()

    dfcatalog = read_data_file(outputfile)
    assert dfcatalog["modelfolder"].to_list() == ["broken", "group/classic3d", "testmodel"]
    assert "#   peak_lum_erg_per_s, peak_time_days: the largest point of light_curve.out" in outputfile.read_text(
        encoding="utf-8"
    )

    rowbroken, rowclassic, rowtestmodel = dfcatalog.iter_rows(named=True)
    assert rowbroken["cells"] is None
    assert rowbroken["timesteps"] is None
    assert captured.err.count("warning: ") == 1
    assert "Model broken in broken: the table gives nan for the cells of model.txt" in captured.err

    assert rowclassic["dimensions"] == 3
    assert rowclassic["cells"] == 1000
    assert rowclassic["timesteps"] == 30
    assert rowclassic["nlte"] is False
    assert rowclassic["nonthermal"] is False
    assert rowclassic["thermalisation"] == "INSTANT_PARTICLE_DEPOSITION=true"
    # MPKTS = 15000 packets for each of the 2 ranks of input.txt
    assert rowclassic["num_packets"] == 30000
    assert rowclassic["artis_version"] == "v2023.10-106-gfcb7e2c"

    # the test model has no artisoptions.h and no log
    assert rowtestmodel["modelname"] == "TEST MODEL"
    assert rowtestmodel["nlte"] is None
    assert rowtestmodel["artis_version"] is None

    for row, folder in ((rowclassic, modelpath_classic_3d), (rowtestmodel, modelpath)):
        # light_curve.out holds the time and the luminosity in Lsun, and an older file repeats the times in a second
        # table, thus the first table ends where the time decreases
        lightcurve = np.loadtxt(folder / "light_curve.out")
        firsttablelength = next(
            (index + 1 for index in range(len(lightcurve) - 1) if lightcurve[index + 1, 0] < lightcurve[index, 0]),
            len(lightcurve),
        )
        times, lums_lsun = lightcurve[:firsttablelength, 0], lightcurve[:firsttablelength, 1]
        assert np.isclose(row["peak_lum_erg_per_s"], lums_lsun.max() * at.constants.Lsun_to_erg_per_s, rtol=1e-6)
        assert np.isclose(row["peak_time_days"], times[lums_lsun.argmax()], rtol=1e-6)


def test_catalog_prints_the_methods_and_the_table_and_asks_before_it_overwrites_a_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The standard output gives the method of each column and the table, and -o asks before it replaces a file."""
    from artistools.catalog import main

    (tmp_path / "testmodel").symlink_to(modelpath, target_is_directory=True)
    main(argsraw=[str(tmp_path)])
    printedlines = capsys.readouterr().out.splitlines()
    # the progress messages come before the product, which is the lines of the methods and then the table
    productstart = printedlines.index("# the method of each column:")
    tablestart = next(index for index, line in enumerate(printedlines) if line.startswith("modelname "))
    assert all(line.startswith("# ") for line in printedlines[productstart:tablestart])
    dfcatalog = pl.read_csv(io.StringIO("\n".join(printedlines[tablestart:])), separator=" ", null_values="nan")
    assert dfcatalog["modelname"].to_list() == ["TEST MODEL"]
    assert dfcatalog["modelfolder"].to_list() == ["testmodel"]

    outputfile = tmp_path / "out" / "catalog.txt"
    main(argsraw=[str(tmp_path), "-o", str(outputfile)])
    assert "# command: " in outputfile.read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        main(argsraw=[str(tmp_path), "-o", str(outputfile)])
    main(argsraw=[str(tmp_path), "-o", str(outputfile), "--force"])
