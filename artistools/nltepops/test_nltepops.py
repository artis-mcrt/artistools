import math
from pathlib import Path

import pandas as pd
import polars as pl

import artistools as at


def test_texifyterm_handles_multiplicity_parity_and_jvalue() -> None:
    assert at.nltepops.texifyterm("o4Fo[2]") == r"$^{4}$F$^{\rm o}_{2}$"
    assert at.nltepops.texifyterm("3P2") == r"$^{3}$P2"


def test_texifyconfiguration_formats_configuration_and_parent_terms() -> None:
    assert at.nltepops.texifyconfiguration("3d6_5D") == r"3d$^{6}$ $^{5}$D"
    assert at.nltepops.texifyconfiguration("3d7(4F)4p_z5G[2]") == r"3d$^{7}$($^{4}$F)4p $^{5}$G$_{2}$"


def test_add_lte_pops_calculates_levels_and_superlevel() -> None:
    ionlevels = pl.DataFrame({"g": [2.0, 4.0, 6.0], "energy_ev": [0.0, 1.0, 2.0]})
    adata = pl.DataFrame({"Z": [26], "ion_stage": [2], "levels": pl.Series([ionlevels], dtype=pl.Object)})

    dfpop = pd.DataFrame({
        "modelgridindex": [0, 0, 0],
        "timestep": [1, 1, 1],
        "Z": [26, 26, 26],
        "ion_stage": [2, 2, 2],
        "level": [0, 1, -1],
        "n_NLTE": [1.0, 0.3, 0.1],
    })

    result = at.nltepops.add_lte_pops(dfpop, adata, [("lte_10000", 10000)], noprint=True)

    k_b = 8.617333262145179e-05
    expected_level1 = 4.0 / 2.0 * math.exp(-(1.0 - 0.0) / k_b / 10000)
    expected_superlevel = 6.0 / 2.0 * math.exp(-(2.0 - 0.0) / k_b / 10000)

    assert math.isclose(result.loc[result["level"] == 0, "lte_10000"].item(), 1.0, rel_tol=1e-12)
    assert math.isclose(result.loc[result["level"] == 1, "lte_10000"].item(), expected_level1, rel_tol=1e-12)
    assert math.isclose(result.loc[result["level"] == 4, "lte_10000"].item(), expected_superlevel, rel_tol=1e-12)


def test_read_files_combines_ranks_and_applies_filters(tmp_path: Path) -> None:
    rank0 = tmp_path / "nlte_0000.out"
    rank1 = tmp_path / "nlte_0001.out"

    rank0.write_text("modelgridindex timestep ionstage level n_NLTE\n7 5 2 0 1.0\n8 5 2 1 2.0\n", encoding="utf-8")
    rank1.write_text("modelgridindex timestep ionstage level n_NLTE\n7 6 2 1 3.0\n7 5 3 0 4.0\n", encoding="utf-8")

    filtered = at.nltepops.read_files(tmp_path, timestep=5, modelgridindex=7)

    assert filtered.height == 2
    assert "ion_stage" in filtered.columns
    assert "ionstage" not in filtered.columns
    assert set(filtered["ion_stage"].to_list()) == {2, 3}
    assert filtered["modelgridindex"].to_list() == [7, 7]
    assert filtered["timestep"].to_list() == [5, 5]
