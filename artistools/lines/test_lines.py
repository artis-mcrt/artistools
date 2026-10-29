import itertools
import math
import re
import typing as t
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import numpy as np
import numpy.typing as npt
import polars as pl
import polars.testing as pltest
import pytest

import artistools as at

modelpath = at.get_path("testdata") / "testmodel"
modelpath_classic_3d = at.get_path("testdata") / "test-classicmode_3d"


def test_ejectaopacity(capsys: pytest.CaptureFixture[str]) -> None:
    """The global Planck mean opacity of the binned expansion opacities agrees with the reference value.

    The opacities need the statistical weights of the levels. A sum without them gives 13.90 cm^2/g.
    """
    at.lines.ejectaopacity.main(
        argsraw=[], modelpath=modelpath, timestep=40, lambdamin=3000.0, lambdamax=4000.0, deltalambda=10.0
    )

    (planckmeanline,) = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("Global Planck mean opacity:")
    ]
    assert np.isclose(float(planckmeanline.split(":")[1].split()[0]), 13.21, rtol=1e-3)


def test_line_contributions_follow_the_level_populations_of_artis() -> None:
    """Each method of the level populations gives the same tau_sobolev and emission as a calculation by hand.

    The test model has NLTE levels and a superlevel for Fe II and Fe III, and LTE populations for Ni II.
    """
    timestep = 10
    dfcells = at.lines.linecontributions.get_line_contributions(
        modelpath, timestep, 3000.0, 30000.0, [(26, 2), (26, 3), (28, 2)], percell=True
    )
    time_s = at.get_timestep_times(modelpath, loc="mid")[timestep] * at.constants.day_to_s
    dfnltepops = at.nltepops.read_nltepops(modelpath, timestep=timestep).filter(pl.col("timestep") == timestep)
    dfestimators = at.scan_estimators(modelpath, timestep=timestep, join_modeldata=True).collect()
    kT_ev = at.constants.K_B_ev_per_K * dfestimators["Te"].item()
    adata = at.atomic.get_levels(modelpath)

    def get_population(
        level: int,
        g: npt.NDArray[np.float64],
        energy_ev: npt.NDArray[np.float64],
        dfionpops: pl.DataFrame,
        nnion: float,
    ) -> float:
        if dfionpops.is_empty():
            boltzmann = g * np.exp(-energy_ev / kT_ev)
            return float(nnion * boltzmann[level] / boltzmann.sum())
        lastnltelevel = dfionpops["level"].max()
        assert isinstance(lastnltelevel, int)
        if level <= lastnltelevel:
            return float(dfionpops.filter(pl.col("level") == level)["n_NLTE"].item())
        superlevel = slice(lastnltelevel + 1, None)
        boltzmann = g[superlevel] * np.exp(-(energy_ev[superlevel] - energy_ev[lastnltelevel + 1]) / kT_ev)
        superlevelpop = dfionpops.filter(pl.col("level") == -1)["n_NLTE"].item()
        return float(superlevelpop * boltzmann[level - lastnltelevel - 1] / boltzmann.sum())

    for ion, Z, ion_stage, source in (
        ("Fe III", 26, 3, "NLTE"),
        ("Fe II", 26, 2, "superlevel"),
        ("Ni II", 28, 2, "LTE"),
    ):
        row = (
            dfcells
            .filter((pl.col("ion") == ion) & (pl.col("n_upper_source") == source))
            .sort("emission_erg_per_s", descending=True)
            .row(0, named=True)
        )
        dflevels = adata.filter((pl.col("Z") == Z) & (pl.col("ion_stage") == ion_stage))["levels"].item()
        levelvalues = (
            dflevels["g"].to_numpy().astype(np.float64),
            dflevels["energy_ev"].to_numpy().astype(np.float64),
            dfnltepops.filter((pl.col("Z") == Z) & (pl.col("ion_stage") == ion_stage)),
            float(dfestimators[f"nnion_{ion.replace(' ', '_')}"].item()),
        )
        g = levelvalues[0]

        lambda_cm = row["lambda_angstroms"] / at.constants.angstroms_per_cm
        n_upper = get_population(row["upper"], *levelvalues)
        tau = (
            row["A_per_s"]
            * lambda_cm**3
            * time_s
            / (8 * math.pi)
            * (g[row["upper"]] / g[row["lower"]] * get_population(row["lower"], *levelvalues) - n_upper)
        )
        emission = (
            n_upper
            * row["A_per_s"]
            * at.constants.h_erg_s
            * at.constants.C_cm_per_s
            / lambda_cm
            * -math.expm1(-tau)
            / tau
            * dfestimators["volume"].item()
        )
        assert np.isclose(row["tau_sobolev"], tau, rtol=1e-6, atol=0.0), ion
        assert np.isclose(row["emission_erg_per_s"], emission, rtol=1e-6, atol=0.0), ion


def test_line_populations_take_the_boltzmann_factor_of_the_ground_level_with_no_superlevel() -> None:
    """For an ion with NLTE rows and no superlevel, a level above the NLTE levels has n_0 g / g_0 exp(-E / k T_exc)."""
    dfcells = pl.DataFrame({"modelgridindex": [0, 1], "T_exc": [5000.0, 10000.0]})
    levels = pl.DataFrame({"level": [0, 1, 2], "g": [2.0, 4.0, 6.0], "energy_ev": [0.0, 1.0, 2.0]})
    dfnltepops = pl.DataFrame({"modelgridindex": [0, 1, 0, 1], "level": [0, 0, 1, 1], "n_NLTE": [10.0, 20.0, 3.0, 4.0]})
    dfpops = at.lines.linecontributions.get_ion_populations(dfcells, levels, [1, 2], dfnltepops, "nnion_X_II")
    dflevel2 = dfpops.filter(pl.col("level") == 2).sort("modelgridindex")
    assert dflevel2["population_source"].unique().to_list() == ["Boltzmann from the ground level"]
    expected = np.array([10.0, 20.0]) * 3.0 * np.exp(-2.0 / at.constants.K_B_ev_per_K / np.array([5000.0, 10000.0]))
    assert np.allclose(dflevel2["population"].to_numpy(), expected, rtol=1e-12, atol=0.0)
    assert dfpops.filter(pl.col("level") == 1).sort("modelgridindex")["population"].to_list() == [3.0, 4.0]


def test_line_summary_gives_the_velocity_of_the_largest_tau_with_a_cell_that_has_no_superlevel_row() -> None:
    """A cell with no superlevel row gave a NaN population, and the summary took the velocity of that NaN tau."""
    from artistools.lines.linecontributions import get_line_summary

    dfcells = pl.DataFrame({"modelgridindex": [0, 1], "T_exc": [5000.0, 5000.0]})
    levels = pl.DataFrame({"level": [0, 1, 2], "g": [2.0, 4.0, 6.0], "energy_ev": [0.0, 1.0, 2.0]})
    dfnltepops = pl.DataFrame({
        "modelgridindex": [0, 1, 0, 1, 0],
        "level": [0, 0, 1, 1, -1],
        "n_NLTE": [10.0, 20.0, 3.0, 4.0, 1.0],
    })
    dfpops = at.lines.linecontributions.get_ion_populations(dfcells, levels, [2], dfnltepops, "nnion_X_II")
    assert dfpops.filter(pl.col("modelgridindex") == 1)["population"].to_list() == [None]

    dfsummary = get_line_summary(
        pl.DataFrame({
            "modelgridindex": [0, 1],
            "velocity_kmps": [1000.0, 2000.0],
            "Te": [5000.0, 5000.0],
            "ion": ["X II", "X II"],
            "lambda_angstroms": [5000.0, 5000.0],
            "lower": [1, 1],
            "upper": [2, 2],
            "A_per_s": [0.1, 0.1],
            "n_lower_source": ["NLTE", "NLTE"],
            "n_upper_source": ["superlevel", "superlevel"],
            "tau_sobolev": [5.0, None],
            "emission_erg_per_s": [1.0, None],
        })
    )
    assert dfsummary["tau_sobolev_max"].item() == 5
    assert dfsummary["velocity_kmps_at_tau_sobolev_max"].item() == 1000
    assert dfsummary["cellcount_tau_sobolev_above_1"].item() == 1


def test_line_populations_of_a_cell_with_no_nlte_row_take_the_lte_populations() -> None:
    """ARTIS writes no NLTE row of an element that fell back to LTE in a cell, and that cell got no populations."""
    dfcells = pl.DataFrame({"modelgridindex": [0, 1], "T_exc": [5000.0, 8000.0], "nnion_X_II": [100.0, 200.0]})
    levels = pl.DataFrame({"level": [0, 1], "g": [2.0, 4.0], "energy_ev": [0.0, 1.0]})
    dfnltepops = pl.DataFrame({"modelgridindex": [0, 0], "level": [0, 1], "n_NLTE": [90.0, 10.0]})
    dfpops = at.lines.linecontributions.get_ion_populations(dfcells, levels, [0, 1], dfnltepops, "nnion_X_II")
    dfcell1 = dfpops.filter(pl.col("modelgridindex") == 1).sort("level")
    assert dfcell1["population_source"].to_list() == ["LTE", "LTE"]
    boltzmann = np.array([2.0, 4.0 * np.exp(-1.0 / at.constants.K_B_ev_per_K / 8000.0)])
    assert np.allclose(dfcell1["population"].to_numpy(), 200.0 * boltzmann / boltzmann.sum(), rtol=1e-12, atol=0.0)
    assert dfpops.filter(pl.col("modelgridindex") == 0).sort("level")["population"].to_list() == [90.0, 10.0]


def test_line_summary_joins_the_cells_of_a_line_with_different_population_methods() -> None:
    """A line with NLTE populations in one cell and LTE populations in another gave two rows of the summary."""
    from artistools.lines.linecontributions import get_line_summary
    from artistools.lines.linecontributions import get_line_summary_of_batches

    def get_cell_table(modelgridindex: int, source: str, emission: float) -> pl.DataFrame:
        return pl.DataFrame({
            "modelgridindex": [modelgridindex],
            "velocity_kmps": [1000.0 * (modelgridindex + 1)],
            "Te": [5000.0],
            "ion": ["Fe II"],
            "lambda_angstroms": [7155.0],
            "lower": [0],
            "upper": [10],
            "A_per_s": [0.1],
            "n_lower_source": [source],
            "n_upper_source": [source],
            "tau_sobolev": [0.5],
            "emission_erg_per_s": [emission],
        })

    dfsummary = get_line_summary_of_batches([
        get_line_summary(get_cell_table(0, "NLTE", 1.0)),
        get_line_summary(get_cell_table(1, "LTE", 3.0)),
    ])
    assert dfsummary.height == 1
    assert dfsummary["n_lower_source"].item() == "LTE/NLTE"
    assert dfsummary["emission_erg_per_s"].item() == 4
    assert np.isclose(dfsummary["velocity_kmps_emission_mean"].item(), 1750.0, rtol=1e-12)


def test_plotlinecontributions_bins_the_cells_of_each_line_by_velocity(
    tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """The emission and the mean tau_sobolev of each bin agree with a sum by hand over the selected cells of the line.

    -costhetamin selects the cells by the direction of their centre, thus the ranking and the emission of each line
    also take only these cells.
    """
    commonargs: dict[str, t.Any] = {"modelpath": modelpath_classic_3d, "lambdamin": 6000.0, "lambdamax": 6400.0}
    nbins = 6
    costhetamin = 0.3
    at.lines.plotlinecontributions.main(
        argsraw=[],
        timestep=25,
        nvelocitybins=nbins,
        maxlines=3,
        rankby="tau",
        costhetamin=costhetamin,
        outputfile=tmp_path / "lines.pdf",
        write_data=True,
        **commonargs,
    )
    dfdrawn = read_data_file(tmp_path / "lines.txt")
    # the rows of the black curve of all the lines come first, and they have no line
    dfalllines = dfdrawn.filter(pl.col("ion").is_null())
    dfdrawn = dfdrawn.filter(pl.col("ion").is_not_null())

    lzmodel, modelmeta = at.scan_modeldata(modelpath_classic_3d)
    dfdirections = (
        at
        .add_derived_cols_to_modeldata(lzmodel, modelmeta=modelmeta)
        .select("modelgridindex", costheta=pl.col("vel_z_mid") / pl.col("vel_r_mid"))
        .collect()
    )
    dfcells = (
        at.lines.linecontributions
        .get_line_contributions(timestep=25, percell=True, **commonargs)
        .join(dfdirections, on="modelgridindex", how="left")
        .filter(pl.col("costheta") >= costhetamin)
    )
    linekeys = list(at.lines.linecontributions.LINEKEYS)
    dfranked = (
        dfcells
        .group_by(linekeys)
        .agg(pl.col("tau_sobolev").max(), pl.col("emission_erg_per_s").sum())
        .sort("tau_sobolev", descending=True)
        .head(3)
    )
    assert dfdrawn.select(linekeys).unique().sort(linekeys).equals(dfranked.select(linekeys).sort(linekeys))

    # the default bins end at the corner of the 3D grid
    binwidth_kmps = modelmeta["vmax_cmps"] / at.constants.km_to_cm * math.sqrt(3) / nbins
    # the black curve sums the emission of all the lines, and the tables did not give it
    dfallbins = (
        dfcells
        .group_by(velocitybin=(pl.col("velocity_kmps") / binwidth_kmps).floor().cast(pl.Int64))
        .agg(pl.col("emission_erg_per_s").sum())
        .sort("velocitybin")
    )
    assert np.allclose(
        dfalllines.sort("x_lower_kmps")["emission_erg_per_s"].to_numpy(),
        dfallbins.filter(pl.col("velocitybin") < nbins)["emission_erg_per_s"].to_numpy(),
        rtol=1e-6,
        atol=0.0,
    )
    for line in dfranked.iter_rows(named=True):
        dflinecells = dfcells.filter(
            (pl.col("ion") == line["ion"]) & (pl.col("lower") == line["lower"]) & (pl.col("upper") == line["upper"])
        ).with_columns(velocitybin=(pl.col("velocity_kmps") / binwidth_kmps).floor().cast(pl.Int64))
        dflinebins = dfdrawn.filter(
            (pl.col("ion") == line["ion"]) & (pl.col("lower") == line["lower"]) & (pl.col("upper") == line["upper"])
        )
        assert dflinebins.height > 1
        for row in dflinebins.iter_rows(named=True):
            dfbincells = dflinecells.filter(pl.col("velocitybin") == round(row["x_lower_kmps"] / binwidth_kmps))
            emission = float(dfbincells["emission_erg_per_s"].sum())
            tau = pl.col("tau_sobolev")
            tau_mean = dfbincells.select((tau * pl.col("volume")).sum() / pl.col("volume").sum()).item()
            assert row["cellcount"] == dfbincells.height
            assert np.isclose(row["emission_erg_per_s"], emission, rtol=1e-6, atol=0.0)
            assert np.isclose(row["tau_sobolev_volume_mean"], tau_mean, rtol=1e-6, atol=0.0)
            assert np.isclose(
                row["emission_fraction_per_kmps"],
                emission / binwidth_kmps / line["emission_erg_per_s"],
                rtol=1e-6,
                atol=0.0,
            )
            # the median is the smallest tau_sobolev whose cumulative volume reaches half of the volume of the bin
            dfsorted = dfbincells.sort("tau_sobolev")
            # equal volumes put half of the volume exactly at the end of a cell, and that tie takes the lower value
            cumulativevolume = dfsorted["volume"].cum_sum().to_numpy()
            halfvolume = 0.5 * cumulativevolume[-1] * (1.0 - 1e-12)
            tau_median = dfsorted["tau_sobolev"][int(np.searchsorted(cumulativevolume, halfvolume))]
            assert np.isclose(row["tau_sobolev_volume_median"], tau_median, rtol=1e-6, atol=0.0)
            volumefraction = float(dfbincells.filter(tau > 1.0)["volume"].sum()) / float(dfbincells["volume"].sum())
            assert np.isclose(row["volume_fraction_tau_sobolev_above_1"], volumefraction, rtol=1e-6, atol=0.0)


@pytest.mark.parametrize(
    ("photospherevelocity", "errortext"),
    [("350000km/s", "above the speed of light"), ("0.05", "has no unit"), ("10km/s", "a third of the width of a cell")],
)
def test_plotlinecontributions_stops_for_a_photosphere_velocity_with_no_unit_or_no_resolution(
    photospherevelocity: str, errortext: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Stop before the estimators for a velocity above c, a number with no unit, or a disk smaller than a cell.

    A bare number took the unit of -velocityunit, whose default depends on the model, thus 0.05 meant for 0.05 c gave
    0.05 km/s.
    """
    with pytest.raises(SystemExit):
        at.lines.plotlinecontributions.main(
            argsraw=[
                *("-modelpath", str(modelpath_classic_3d), "-timestep", "25"),
                *("-lambdamin", "6000", "-lambdamax", "6400", "-velocityunit", "kmps"),
                *("-troughdirbin", "90", "-photospherevelocity", photospherevelocity),
                *("-o", str(tmp_path / "lines.pdf")),
            ]
        )
    captured = capsys.readouterr()
    assert errortext in captured.out + captured.err


def test_plotlinecontributions_reports_an_arrival_time_outside_the_timesteps(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An arrival time before the first timestep or after the last one is outside each timestep.

    The log clipped such a time to the first or the last timestep, but the packet commands keep no packet there.
    """
    from artistools.lines.linecontributions import LineVelocityProfiles

    tstarts = at.get_timestep_times(modelpath_classic_3d, loc="start")
    tends = at.get_timestep_times(modelpath_classic_3d, loc="end")
    dfrankedcells = pl.DataFrame({"velocity_kmps": [0.5 * at.constants.C_cm_per_s / at.constants.km_to_cm]})
    profiles = LineVelocityProfiles(
        pl.DataFrame(),
        pl.DataFrame(),
        pl.DataFrame(),
        pl.DataFrame(),
        dfrankedcells.with_columns(emission_erg_per_s=pl.lit(1.0)),
    )
    at.lines.plotlinecontributions.print_arrival_times(modelpath_classic_3d, profiles, tstarts[0] * 1.01, "kmps")
    assert "the first time in no timestep, because it is before the start" in capsys.readouterr().out
    at.lines.plotlinecontributions.print_arrival_times(modelpath_classic_3d, profiles, tends[-1] * 0.99, "kmps")
    assert "the last time in no timestep, because it is after the end" in capsys.readouterr().out


def test_sobolev_trough_of_a_shell_agrees_with_the_analytic_depth() -> None:
    """A shell of constant tau_sobolev from 0.9 v_ph to 0.9 vmax gives the depth that the geometry gives.

    For a line-of-sight velocity u above v_ph, each ray in front of the disk crosses the shell, thus the depth is
    1 - exp(-tau). For u below v_ph, only the ring of the disk with an impact parameter above sqrt(v_ph^2 - u^2) is in
    front, thus the depth is (1 - exp(-tau)) u^2 / v_ph^2, as a mean over the bin. A shell gives the same depth for
    each direction. On a coarse grid with bins narrower than a cell, a cell covers each bin that it crosses. The
    earlier code put the whole volume of a cell in the bin of its centre, and the depths then went from 0.24 to 1.11.

    Bins that end below the photosphere, e.g. with -velocitymax, gave no plateau. One bin gave a blend above zero at
    only one wavelength, because the blend went to zero at the centre of the outer bins. The wavelength grid of the
    blend went to the fastest part of a cell, although the bins above the shell had no depth.
    """
    from artistools.lines.linecontributions import CellLineInputs
    from artistools.lines.linecontributions import get_sobolev_trough
    from artistools.lines.linecontributions import LineVelocityProfiles

    vmax_kmps, vph_kmps, tau, time_days = 1.0e4, 4000.0, 2.0, 1.0
    line = {"ion": "X I", "lambda_angstroms": 5000.0, "lower": 0, "upper": 1, "A_per_s": 1.0}
    absorbed = -math.expm1(-tau)

    def get_shell_trough(
        ncells: int, edges: list[float], dirbin: int, *, relativisticdoppler: bool
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        cellwidth_kmps = 2 * vmax_kmps / ncells
        centres = -vmax_kmps + cellwidth_kmps * (np.arange(ncells) + 0.5)
        vx, vy, vz = (axis.ravel() for axis in np.meshgrid(centres, centres, centres, indexing="ij"))
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        cellvolume = (cellwidth_kmps * at.constants.km_to_cm * time_days * at.constants.day_to_s) ** 3
        modelgridindex = np.arange(ncells**3)
        dfrankedcells = pl.DataFrame({
            **{key: [value] * ncells**3 for key, value in line.items()},
            "modelgridindex": modelgridindex,
            "volume": np.full(ncells**3, cellvolume),
            # the shell starts inside the photosphere, thus each part of a cell outside the photosphere has tau
            "tau_sobolev": np.where((speed > 0.9 * vph_kmps) & (speed < 0.9 * vmax_kmps), tau, 0.0),
        })
        dfcells = pl.DataFrame({
            "modelgridindex": modelgridindex,
            **{
                f"vel_{axis}_mid": values * at.constants.km_to_cm
                for axis, values in zip("xyz", (vx, vy, vz), strict=True)
            },
        })
        profiles = LineVelocityProfiles(
            pl.DataFrame(), pl.DataFrame([line]), pl.DataFrame(), pl.DataFrame(), dfrankedcells
        )
        inputs = CellLineInputs(dfcells, pl.DataFrame(), {}, pl.DataFrame(), time_days)
        return get_sobolev_trough(
            profiles,
            inputs,
            dirbin,
            vph_kmps,
            edges,
            (cellwidth_kmps, cellwidth_kmps, cellwidth_kmps),
            relativisticdoppler=relativisticdoppler,
            averageoverphi=True,
        )

    edges = [1000.0 * index for index in range(9)]
    expected = [
        absorbed * (upper**3 - lower**3) / (3 * (upper - lower) * vph_kmps**2) if upper <= vph_kmps else absorbed
        for lower, upper in itertools.pairwise(edges)
    ]
    for dirbin, relativisticdoppler in ((90, True), (50, False)):
        dfdepths, dfblend = get_shell_trough(64, edges, dirbin, relativisticdoppler=relativisticdoppler)
        assert np.allclose(dfdepths["trough_depth"].to_numpy()[: len(expected)], expected, rtol=0.0, atol=0.01), dirbin
        deepest = dfblend.top_k(1, by="trough_depth").row(0, named=True)
        assert np.isclose(deepest["trough_depth"], absorbed, rtol=0.01, atol=0.0)
        # the plateau from 4000 to 8000 km/s gives the wavelengths of the Doppler shift of 5000 Å
        assert 4866.0 < deepest["lambda_angstroms"] < 4934.0

    # a cell of the 10^3 grid is 2000 km/s wide, and each bin is 400 km/s wide. Near the z axis the shell of these
    # cells ends at the cell face at z = 8000 km/s, thus the test takes the bins from 4400 to 7200 km/s
    dfdepths, _ = get_shell_trough(10, [400.0 * index for index in range(21)], 90, relativisticdoppler=False)
    dfplateau = dfdepths.filter(pl.col("x_lower_kmps").is_between(4400.0, 6800.0))
    assert np.allclose(dfplateau["trough_depth"].to_numpy(), absorbed, rtol=0.0, atol=0.02)

    # the bins continue past the last edge with the same width, thus the plateau above the photosphere stays
    dfshort, dfshortblend = get_shell_trough(64, edges[:4], 90, relativisticdoppler=False)
    assert np.allclose(dfshort["trough_depth"].to_numpy()[: len(expected)], expected, rtol=0.0, atol=0.01)
    assert np.isclose(dfshortblend["trough_depth"].to_numpy().max(), absorbed, rtol=0.01, atol=0.0)
    # the shell ends at 9000 km/s, and the blend ends one bin after the last depth. The grid went to the fastest part
    # of a cell at 17 000 km/s, and the 400 points then covered a range twice as wide
    c_kmps = at.constants.C_cm_per_s / at.constants.km_to_cm
    shortestlambda_angstroms = float(dfshortblend["lambda_angstroms"].to_numpy().min())
    assert 5000.0 * (1 - 11000.0 / c_kmps) < shortestlambda_angstroms < 5000.0 * (1 - 9000.0 / c_kmps)

    # the blend samples the rays at each wavelength, thus one wide bin still gives the plateau of the shell
    dfonebin, dfonebinblend = get_shell_trough(10, [0.0, 2.0 * vmax_kmps], 90, relativisticdoppler=False)
    assert dfonebin.height == 1
    assert np.isclose(dfonebinblend["trough_depth"].to_numpy().max(), absorbed, rtol=0.0, atol=0.07)


def get_trough_of_box_grid(
    gridshape: tuple[int, int, int],
    tauofcell: "Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]",
    nlines: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the trough of nlines lines at 5000 Å with the same tau_sobolev, on a grid from -10000 to 10000 km/s.

    tauofcell gives the tau_sobolev of each cell from the velocities of the cell centres [km/s].
    """
    from artistools.lines.linecontributions import CellLineInputs
    from artistools.lines.linecontributions import get_sobolev_trough
    from artistools.lines.linecontributions import LineVelocityProfiles

    vmax_kmps = 1.0e4
    widths = (2 * vmax_kmps / gridshape[0], 2 * vmax_kmps / gridshape[1], 2 * vmax_kmps / gridshape[2])
    centres = [-vmax_kmps + width * (np.arange(ncells) + 0.5) for ncells, width in zip(gridshape, widths, strict=True)]
    vx, vy, vz = (axis.ravel() for axis in np.meshgrid(*centres, indexing="ij"))
    modelgridindex = np.arange(len(vx))
    lines = [
        {"ion": "X I", "lambda_angstroms": 5000.0, "lower": 0, "upper": upper, "A_per_s": 1.0}
        for upper in range(1, nlines + 1)
    ]
    dfrankedcells = pl.concat([
        pl.DataFrame({
            **{key: [value] * len(vx) for key, value in line.items()},
            "modelgridindex": modelgridindex,
            "volume": np.ones(len(vx)),
            "tau_sobolev": tauofcell(vx, vy, vz),
        })
        for line in lines
    ])
    dfcells = pl.DataFrame({
        "modelgridindex": modelgridindex,
        **{f"vel_{axis}_mid": values * at.constants.km_to_cm for axis, values in zip("xyz", (vx, vy, vz), strict=True)},
    })
    profiles = LineVelocityProfiles(pl.DataFrame(), pl.DataFrame(lines), pl.DataFrame(), pl.DataFrame(), dfrankedcells)
    inputs = CellLineInputs(dfcells, pl.DataFrame(), {}, pl.DataFrame(), 1.0)
    edges = [1000.0 * index for index in range(9)]
    return get_sobolev_trough(
        profiles, inputs, 90, 4000.0, edges, widths, relativisticdoppler=False, averageoverphi=True
    )


def test_sobolev_trough_blends_lines_on_each_ray_and_takes_each_cell_width() -> None:
    """Two lines that absorb the same half of the disk remove half of its flux, and a cell has a width on each axis.

    The blend was 1 - prod(1 - depth) of the depths over the disk, which gave 0.75 for two saturated lines on the same
    half of the disk. The rays also took the x width of a cell for each axis.
    """
    tau = 50.0

    def get_halfshell_tau(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray) -> np.ndarray:
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        return np.where((speed > 3600.0) & (speed < 9000.0) & (vx > 0.0), tau, 0.0)

    dfdepths, dfblend = get_trough_of_box_grid((48, 48, 48), get_halfshell_tau, nlines=2)
    dfplateau = dfdepths.filter(pl.col("x_lower_kmps").is_between(4000.0, 6000.0))
    assert np.allclose(dfplateau["trough_depth"].to_numpy(), 0.5, rtol=0.0, atol=0.03)
    assert np.isclose(float(dfblend["trough_depth"].to_numpy().max()), 0.5, rtol=0.0, atol=0.03)

    def get_shell_tau(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray) -> np.ndarray:
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        return np.where((speed > 3600.0) & (speed < 9000.0), 2.0, 0.0)

    # the cells are twice as wide along z as along x and y
    dfdepths, _ = get_trough_of_box_grid((64, 64, 32), get_shell_tau, nlines=1)
    dfplateau = dfdepths.filter(pl.col("x_lower_kmps").is_between(4000.0, 6000.0))
    assert np.allclose(dfplateau["trough_depth"].to_numpy(), -math.expm1(-2.0), rtol=0.0, atol=0.02)


def test_linecontributions_taumin_keeps_each_line_that_reaches_it() -> None:
    """-taumin leaves out only the lines whose tau_sobolev stays below it in each cell.

    The upper bound of tau_sobolev is never below the tau_sobolev of a cell, thus -taumin keeps each line that reaches
    it.
    """
    commonargs: dict[str, t.Any] = {"modelpath": modelpath_classic_3d, "lambdamin": 3000.0, "lambdamax": 5000.0}
    linekeys = list(at.lines.linecontributions.LINEKEYS)
    dfall = at.lines.linecontributions.get_line_contributions(timestep=25, **commonargs)
    dfkept = at.lines.linecontributions.get_line_contributions(timestep=25, taumin=1.0, **commonargs)
    reachingkeys = set(dfall.filter(pl.col("tau_sobolev_max") >= 1.0).select(linekeys).iter_rows())
    keptkeys = set(dfkept.select(linekeys).iter_rows())
    assert reachingkeys
    assert reachingkeys <= keptkeys
    assert len(keptkeys) < dfall.height


def test_taumin_keeps_a_line_whose_lower_level_is_above_the_nlte_levels() -> None:
    """The -taumin bound of an ion with NLTE rows takes n_l <= n_ion max(1, g_l / g_0).

    A level above the NLTE levels of an ion with no superlevel takes n_0 (g_l / g_0) exp(-(E_l - E_0) / k T_exc), which
    can be above n_ion. The earlier bound n_l <= n_ion left out such a line, although its tau_sobolev reached -taumin.
    """
    from artistools.lines.linecontributions import get_cell_line_table
    from artistools.lines.linecontributions import get_lines_that_can_reach_tau
    from artistools.lines.linecontributions import NLTEPOPS_SCHEMA

    dfcells = pl.DataFrame({
        "modelgridindex": [0],
        "velocity_kmps": [5000.0],
        "volume": [1e45],
        "Te": [1e4],
        "nne": [1e8],
        "T_exc": [1e4],
        "nnion_Fe_II": [1.0],
    })
    levelsofion = {
        (26, 2): pl.DataFrame({
            "level": [0, 1, 2],
            "g": [1.0, 9.0, 5.0],
            "energy_ev": [0.0, 0.01, 2.0],
            "levelname": ["a", "b", "c"],
        })
    }
    dflines = pl.DataFrame({
        "Z": [26],
        "ion_stage": [2],
        "ion": ["Fe II"],
        "lambda_angstroms": [6230.4],
        "lower": [1],
        "upper": [2],
        "A_per_s": [1e4],
        "lower_g": [9.0],
        "upper_g": [5.0],
    })
    # an NLTE row for the ground level and no superlevel row
    dfnltepops = pl.DataFrame(
        {"modelgridindex": [0], "Z": [26], "ion_stage": [2], "level": [0], "n_NLTE": [0.9]}, schema=NLTEPOPS_SCHEMA
    )
    dfcelllines = get_cell_line_table(dfcells, dflines, levelsofion, dfnltepops, time_days=1.0)
    assert dfcelllines["n_lower_per_cm3"].item() > dfcells["nnion_Fe_II"].item()
    tau = dfcelllines["tau_sobolev"].item()
    dfkept = get_lines_that_can_reach_tau(dflines, dfcells, levelsofion, {(26, 2)}, tau, time_days=1.0)
    assert dfkept.height == 1

    # a second cell with a NaN estimator gave a bound of NaN, and the bound then left out the line
    dfwithnan = pl.concat([
        dfcells,
        dfcells.with_columns(modelgridindex=pl.col("modelgridindex") + 1, nnion_Fe_II=pl.lit(math.nan)),
    ])
    assert get_lines_that_can_reach_tau(dflines, dfwithnan, levelsofion, {(26, 2)}, tau, time_days=1.0).height == 1


def test_linecontributions_takes_the_nlte_levels_of_each_cell() -> None:
    """A level that only a different cell resolves takes the superlevel population of this cell.

    The highest NLTE level was the maximum over all the cells. A cell that resolves fewer levels then got a null
    population for a level of its superlevel, and the line had no tau_sobolev in that cell.
    """
    from artistools.lines.linecontributions import get_cell_line_table
    from artistools.lines.linecontributions import NLTEPOPS_SCHEMA

    dfcells = pl.DataFrame({
        "modelgridindex": [0, 1],
        "velocity_kmps": [5000.0, 6000.0],
        "volume": [1e45, 1e45],
        "Te": [1e4, 1e4],
        "nne": [1e8, 1e8],
        "T_exc": [1e4, 1e4],
        "nnion_Fe_II": [1.0, 1.0],
    })
    levelsofion = {
        (26, 2): pl.DataFrame({
            "level": [0, 1, 2, 3],
            "g": [1.0, 3.0, 5.0, 7.0],
            "energy_ev": [0.0, 0.5, 1.0, 3.0],
            "levelname": ["a", "b", "c", "d"],
        })
    }
    dflines = pl.DataFrame({
        "Z": [26],
        "ion_stage": [2],
        "ion": ["Fe II"],
        "lambda_angstroms": [6200.0],
        "lower": [2],
        "upper": [3],
        "A_per_s": [1e4],
        "lower_g": [5.0],
        "upper_g": [7.0],
    })
    # cell 0 resolves levels 0 and 1, and cell 1 resolves levels 0 to 2. Each cell has a superlevel row (level -1)
    dfnltepops = pl.DataFrame(
        {
            "modelgridindex": [0, 0, 0, 1, 1, 1, 1],
            "Z": [26] * 7,
            "ion_stage": [2] * 7,
            "level": [0, 1, -1, 0, 1, 2, -1],
            "n_NLTE": [0.5, 0.2, 0.3, 0.5, 0.2, 0.2, 0.1],
        },
        schema=NLTEPOPS_SCHEMA,
    )
    dfcelllines = get_cell_line_table(dfcells, dflines, levelsofion, dfnltepops, time_days=1.0).sort("modelgridindex")
    assert dfcelllines["n_lower_source"].to_list() == ["superlevel", "NLTE"]
    assert dfcelllines["n_upper_source"].to_list() == ["superlevel", "superlevel"]
    assert dfcelllines["n_lower_per_cm3"].null_count() == 0
    assert np.isclose(dfcelllines["n_lower_per_cm3"][1], 0.2, rtol=1e-12, atol=0.0)
    # in cell 0, the superlevel holds levels 2 and 3, thus level 2 takes its Boltzmann share of 0.3
    assert 0.0 < dfcelllines["n_lower_per_cm3"][0] < 0.3


def test_linecontributions_writes_a_file_only_with_the_output_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command wrote linecontributions.txt into the working folder with no -o, because -o had a default name."""
    monkeypatch.chdir(tmp_path)
    commonargs: dict[str, t.Any] = {"modelpath": modelpath, "timestep": 10, "lambdamin": 4600.0, "lambdamax": 4700.0}
    at.lines.linecontributions.main(argsraw=[], **commonargs)
    assert not list(tmp_path.iterdir())
    assert '"Fe III" 4659.' in capsys.readouterr().out
    # the skill gives the model path as the last positional argument, and the parser rejected it
    at.lines.linecontributions.main(
        argsraw=["-timestep", "10", "-lambdamin", "4600", "-lambdamax", "4700", str(modelpath)]
    )
    assert '"Fe III" 4659.' in capsys.readouterr().out

    outputfile = tmp_path / "lines.txt"
    at.lines.linecontributions.main(argsraw=[], outputfile=outputfile, **commonargs)
    filetext = outputfile.read_text(encoding="utf-8")
    assert "# column tau_sobolev_max: the maximum Sobolev optical depth of the line in the cells" in filetext


@mock.patch.object(mplax.Axes, "axhline", side_effect=mplax.Axes.axhline, autospec=True)
@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_plotopacity_draws_ratios_and_the_planck_mean(
    mockplot: mock.MagicMock, mockaxhline: mock.MagicMock, tmp_path: Path
) -> None:
    """The plot has the bins and a moving average of each opacity, a dashed capped opacity, ratios, and the Planck mean.

    The Planck mean takes the bins of the x range, with the Planck function at the temperature of the cell. Only
    --showplanckmean draws it.
    """
    at.lines.plotopacity.main(
        argsraw=[
            "-modelpath",
            str(modelpath),
            "-timestep",
            "40",
            "-xmin",
            "3000",
            "-xmax",
            "4000",
            "--showplanckmean",
            "-o",
            str(tmp_path / "opac.pdf"),
        ]
    )
    axes = [call.args[0] for call in mockplot.call_args_list]
    assert len(set(axes)) == 2, "the opacities and the ratios need two frames"
    mainplots = [call for call in mockplot.call_args_list if call.args[0] is axes[0]]
    assert len(mainplots) == 6, "each opacity needs its bins and its moving average"
    assert [call.kwargs["linestyle"] for call in mainplots if call.kwargs.get("label")] == ["-", "--", "-"]

    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    dfcell = at.lines.core.get_cell_estimators(modelpath, timestep, None, "Te")
    edges = at.lines.core.get_lambda_bin_edges(3000.0, 4000.0, 20.0)
    lines = at.lines.core.get_opacity_lines(
        at.lines.core.get_opacity_atomic_data(modelpath), dfcell.columns, edges, time_days
    )
    dfbins = at.lines.core.get_expansion_opacities(lines, dfcell, edges, time_days)
    lambda_cm = dfbins["lambda_angstroms_bin_mid"].to_numpy() * 1e-8
    temperature = dfcell["T_exc"].item()
    planck = lambda_cm**-5 / np.expm1(
        at.constants.h_erg_s * at.constants.C_cm_per_s / lambda_cm / temperature / at.constants.K_B_erg_per_K
    )
    expected = float(np.sum(planck * dfbins["exopac"].to_numpy()) / np.sum(planck))
    assert mockaxhline.call_count == 1
    assert np.isclose(mockaxhline.call_args.args[1], expected, rtol=1e-9, atol=0.0)

    # the line and its calculation need --showplanckmean
    at.lines.plotopacity.main(
        argsraw=[
            "-modelpath",
            str(modelpath),
            "-timestep",
            "40",
            "-xmin",
            "3000",
            "-xmax",
            "4000",
            "-o",
            str(tmp_path / "noplanck.pdf"),
        ]
    )
    assert mockaxhline.call_count == 1


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_plotopacity_draws_each_cap_and_the_line_count(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """-taucaps draws an opacity for each cap, and --showlinecount adds a panel with the number of lines in each bin.

    The panel counts each line that the opacities sum. A cap that is given two times gives one opacity. The :g format
    gave 1.0000001 the column name of 1, thus the kernel stopped with an error.
    """
    at.lines.plotopacity.main(
        argsraw=[
            "-modelpath",
            str(modelpath),
            "-timestep",
            "40",
            "-xmin",
            "3000",
            "-xmax",
            "4000",
            "-taucaps",
            "0.1",
            "1",
            "10",
            "1",
            "1.0000001",
            "--showlinecount",
            "-o",
            str(tmp_path / "opac.pdf"),
        ]
    )
    axes = list(dict.fromkeys(call.args[0] for call in mockplot.call_args_list))
    assert len(axes) == 3, "the opacities, the ratios, and the number of lines need three panels"
    labels = [call.kwargs["label"] for call in mockplot.call_args_list if call.kwargs.get("label")]
    assert labels == [
        "Expansion opacity",
        *(rf"Line-binned, $\tau_\mathrm{{l,max}}$ = {taucap}" for taucap in ("0.1", "1", "10", "1.0000001")),
        "Line-binned",
    ]
    ratioplots = [call for call in mockplot.call_args_list if call.args[0] is axes[1]]
    assert len(ratioplots) == 5, "each line-binned opacity needs a ratio"

    (linecountplot,) = [call for call in mockplot.call_args_list if call.args[0] is axes[2]]
    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    dfcell = at.lines.core.get_cell_estimators(modelpath, timestep, None, "Te")
    edges = at.lines.core.get_lambda_bin_edges(3000.0, 4000.0, 20.0)
    lines = at.lines.core.get_opacity_lines(
        at.lines.core.get_opacity_atomic_data(modelpath), dfcell.columns, edges, time_days
    )
    # each bin draws its count at its lower edge and at its upper edge
    assert np.nansum(linecountplot.args[2]) == 2 * lines.dflines.height


def test_plotopacity_average_cell_takes_the_mean_composition(capsys: pytest.CaptureFixture[str]) -> None:
    """--averagecell takes one cell with the mass-weighted mean of n_ion / rho, of rho, and of T_exc, and logs T_exc.

    A model of one cell gives the same cell. A cell with no temperature does not count in the mean temperature.
    """
    dfone = at.lines.core.get_cell_estimators(modelpath, 40, None, "Te")
    pltest.assert_frame_equal(
        at.lines.plotopacity.get_average_cell(dfone),
        dfone.select(at.lines.plotopacity.get_average_cell(dfone).columns),
        check_dtypes=False,
        rel_tol=1e-12,
    )

    dfcells = pl.DataFrame({
        "modelgridindex": [0, 1, 2],
        "timestep": [5, 5, 5],
        "T_exc": [1000.0, 3000.0, None],
        "rho": [1.0, 2.0, 4.0],
        "mass_g": [1.0, 3.0, 4.0],
        "nnion_Fe_II": [2.0, 4.0, 8.0],
    })
    capsys.readouterr()
    dfmean = at.lines.plotopacity.get_average_cell(dfcells)
    assert np.isclose(dfmean["T_exc"].item(), (1000.0 + 3 * 3000.0) / 4, rtol=1e-12, atol=0.0)
    meanrho = (1.0 + 3 * 2.0 + 4 * 4.0) / 8
    assert np.isclose(dfmean["rho"].item(), meanrho, rtol=1e-12, atol=0.0)
    assert np.isclose(dfmean["nnion_Fe_II"].item(), meanrho * (2.0 + 3 * 2.0 + 4 * 2.0) / 8, rtol=1e-12, atol=0.0)
    assert "T_exc = 2500 K" in capsys.readouterr().out
    assert (
        at.lines.plotopacity.get_cells_text(None, None, None, "TJ = 2500 K")
        == "mean composition of all cells at TJ = 2500 K"
    )


def test_excitation_temperature_is_the_temperature_that_artis_used(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """-exctemperature auto takes TJ or Te from LTEPOP_EXCITATION_USE_TJ of the model, and the log names it.

    The kernel took Te before. A classic model sets LTEPOP_EXCITATION_USE_TJ = true, and after the LTE timesteps its Te
    differs from TJ, thus the level populations did not match ARTIS. A commented line does not count.
    """
    get_column = at.lines.core.get_excitation_temperature_column
    capsys.readouterr()
    assert get_column(tmp_path, "auto") == "Te"
    assert "gives no LTEPOP_EXCITATION_USE_TJ" in capsys.readouterr().err

    (tmp_path / "artis").mkdir()
    optionspath = tmp_path / "artis" / "artisoptions.h"
    for value, expectedcolumn in (("true", "TJ"), ("false", "Te")):
        optionspath.write_text(
            f"// constexpr bool LTEPOP_EXCITATION_USE_TJ = {'false' if value == 'true' else 'true'};\n"
            f"constexpr bool LTEPOP_EXCITATION_USE_TJ = {value};\n",
            encoding="utf-8",
        )
        assert get_column(tmp_path, "auto") == expectedcolumn
        assert f"T_exc = {expectedcolumn}," in capsys.readouterr().out
    assert get_column(tmp_path, "TJ") == "TJ"
    assert "-exctemperature TJ" in capsys.readouterr().out


def test_expansion_opacities_keep_the_values_of_the_join_query() -> None:
    """The Rust kernel gives the values of the earlier polars query.

    The earlier query joined each cell with each line, and the reference sums come from it. Only the order
    of the additions is different, thus the values agree within the rounding error.
    """
    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    dfcell = at.lines.core.get_cell_estimators(modelpath, timestep, None, "Te")
    lambda_bin_edges = at.lines.core.get_lambda_bin_edges(3000.0, 4000.0, 10.0)
    opacitylines = at.lines.core.get_opacity_lines(
        at.lines.core.get_opacity_atomic_data(modelpath), dfcell.columns, lambda_bin_edges, time_days
    )

    dfopacities = at.lines.core.get_expansion_opacities(opacitylines, dfcell, lambda_bin_edges, time_days)

    assert dfopacities.height == 100
    for column, expectedsum in {
        "exopac": 1397.6901658607103,
        "linebinned": 11675.7786539805,
        "linebinned_cap1": 1652.4668052744682,
    }.items():
        assert math.isclose(dfopacities[column].sum(), expectedsum, rel_tol=1e-12), column


def test_expansion_opacities_give_each_cap_its_own_column() -> None:
    """The kernel gives the sum of each cap to the column of that cap. The order of the caps has no effect.

    A cap of 1 gives the value of the regression test above. A larger cap gives a larger sum, and a cap above each
    tau_sobolev gives the sum with no cap.
    """
    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    dfcell = at.lines.core.get_cell_estimators(modelpath, timestep, None, "Te")
    lambda_bin_edges = at.lines.core.get_lambda_bin_edges(3000.0, 4000.0, 10.0)
    opacitylines = at.lines.core.get_opacity_lines(
        at.lines.core.get_opacity_atomic_data(modelpath), dfcell.columns, lambda_bin_edges, time_days
    )

    taucaps = (10.0, 1e300, 0.1, 1.0)
    dfopacities = at.lines.core.get_expansion_opacities(opacitylines, dfcell, lambda_bin_edges, time_days, taucaps)

    sums = {taucap: dfopacities[at.lines.core.get_capped_column(taucap)].sum() for taucap in sorted(taucaps)}
    assert at.lines.core.get_capped_column(1234567.0) != at.lines.core.get_capped_column(1234568.0)
    assert math.isclose(sums[1.0], 1652.4668052744682, rel_tol=1e-12)
    assert math.isclose(sums[1e300], dfopacities["linebinned"].sum(), rel_tol=1e-12)
    assert list(sums.values()) == sorted(sums.values())
    assert len(set(sums.values())) == len(sums)


def test_expansion_opacities_of_a_null_population_are_zero() -> None:
    """A null population of an ion gives no opacity from that ion, and a null temperature gives no opacity.

    The kernel reads contiguous columns with no nulls, thus get_expansion_opacities() must replace each null.
    """
    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    dfcell = at.lines.core.get_cell_estimators(modelpath, timestep, None, "Te")
    lambda_bin_edges = at.lines.core.get_lambda_bin_edges(3000.0, 4000.0, 10.0)
    opacitylines = at.lines.core.get_opacity_lines(
        at.lines.core.get_opacity_atomic_data(modelpath), dfcell.columns, lambda_bin_edges, time_days
    )
    ionstr = opacitylines.ionstrs[0]

    def get_opacities(dfcells: pl.DataFrame) -> pl.DataFrame:
        return at.lines.core.get_expansion_opacities(opacitylines, dfcells, lambda_bin_edges, time_days).select(
            at.lines.core.get_opacity_columns(at.lines.core.DEFAULT_TAUCAPS)
        )

    pltest.assert_frame_equal(
        get_opacities(dfcell.with_columns(pl.lit(None, dtype=pl.Float32).alias(f"nnion_{ionstr}"))),
        get_opacities(dfcell.with_columns(pl.lit(0.0, dtype=pl.Float32).alias(f"nnion_{ionstr}"))),
    )
    dfnotemperature = get_opacities(dfcell.with_columns(pl.lit(None, dtype=pl.Float32).alias("T_exc")))
    assert np.allclose(dfnotemperature.select(pl.all().abs().max()).row(0), 0.0, rtol=0.0, atol=0.0)


def test_expansion_opacities_keep_a_nan_in_each_sum() -> None:
    """A NaN temperature gives NaN level populations, and each of the sums must then be NaN.

    f64::min(NaN, 1) is 1, and NaN.abs() >= 1e-18 is false, thus the kernel gave a finite capped opacity
    and a finite exopac for such a cell.
    """
    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    dfcell = at.lines.core.get_cell_estimators(modelpath, timestep, None, "Te").with_columns(
        pl.lit(float("nan"), dtype=pl.Float32).alias("T_exc")
    )
    lambda_bin_edges = at.lines.core.get_lambda_bin_edges(3000.0, 4000.0, 10.0)
    opacitylines = at.lines.core.get_opacity_lines(
        at.lines.core.get_opacity_atomic_data(modelpath), dfcell.columns, lambda_bin_edges, time_days
    )

    taucaps = (0.1, 1.0)
    dfopacities = at.lines.core.get_expansion_opacities(opacitylines, dfcell, lambda_bin_edges, time_days, taucaps)

    opacitycolumns = at.lines.core.get_opacity_columns(taucaps)
    isnan = dfopacities.select(pl.col(opacitycolumns).is_nan())
    assert isnan["linebinned"].any()
    for column in opacitycolumns:
        assert isnan[column].equals(isnan["linebinned"]), column


def test_expansion_opacities_skip_a_line_with_a_null_constant() -> None:
    """A line with a null A value adds nothing, as a line that the atomic data does not hold.

    The kernel reads columns with no nulls, thus a null constant stopped the sum with an error.
    """
    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    dfcell = at.lines.core.get_cell_estimators(modelpath, timestep, None, "Te")
    lambda_bin_edges = at.lines.core.get_lambda_bin_edges(3000.0, 4000.0, 10.0)
    adata = at.lines.core.get_opacity_atomic_data(modelpath)
    isfirstline = pl.int_range(pl.len()) == 0

    def get_opacities(transitions: list[pl.LazyFrame]) -> pl.DataFrame:
        adatachanged = adata.with_columns(pl.Series("transitions", transitions, dtype=pl.Object))
        opacitylines = at.lines.core.get_opacity_lines(adatachanged, dfcell.columns, lambda_bin_edges, time_days)
        return at.lines.core.get_expansion_opacities(opacitylines, dfcell, lambda_bin_edges, time_days)

    dfnullconstant = get_opacities([
        dftransitions.lazy().with_columns(A=pl.when(~isfirstline).then(pl.col("A")))
        for dftransitions in adata["transitions"]
    ])
    dfnoline = get_opacities([dftransitions.lazy().filter(~isfirstline) for dftransitions in adata["transitions"]])

    pltest.assert_frame_equal(dfnullconstant, dfnoline)


def test_lambda_bin_edges_reject_a_range_with_no_bin() -> None:
    """A range with no bin stops with a message, not with an IndexError in get_expansion_opacities()."""
    with pytest.raises(ValueError, match="holds no bin"):
        at.lines.core.get_lambda_bin_edges(5000.0, 4000.0, 10.0)


def test_lambda_bin_edges_cover_the_full_range() -> None:
    """The bins cover the full wavelength range, also when the division of the range rounds down.

    (4000 - 3000) / 0.1 is 9999.999999999998, and int() of it gave 9999 bins. Thus the last bin was missing.
    A range that does not hold a whole number of bins lost the part after the last whole bin.
    """
    edges = at.lines.core.get_lambda_bin_edges(3000.0, 4000.0, 0.1)
    assert len(edges) == 10001
    assert math.isclose(edges[-1], 4000.0, rel_tol=1e-12)

    edges = at.lines.core.get_lambda_bin_edges(1000.0, 25010.0, 20.0)
    assert len(edges) == 1202
    assert math.isclose(edges[-1], 25020.0, rel_tol=1e-12)


def test_plotopacity_weights_the_cells_by_mass() -> None:
    """The mean over the cells weights each cell by its mass, and it sums the cells of every batch.

    Cell k holds k times the ion populations and k times the mass of the test cell. The line-binned
    opacity is linear in the populations. Thus, for n cells, the mean is sum(k^2) / sum(k) = (2n + 1) / 3
    times the opacity of the test cell. The first CELLSPERBATCH cells fill one batch, and the last 8 cells
    go into a second batch.
    """
    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    adata = at.lines.core.get_opacity_atomic_data(modelpath)
    dfcell = at.lines.core.get_cell_estimators(modelpath, timestep, None, "Te")
    assert dfcell.height == 1

    cellcount = at.lines.core.CELLSPERBATCH + 8
    dfcells = pl.concat([
        dfcell.with_columns(
            pl.lit(k - 1, dtype=dfcell.schema["modelgridindex"]).alias("modelgridindex"),
            pl.col("mass_g") * k,
            # a population is Float32, and k times it rounds in Float32
            pl.col("^nnion_.*$").cast(pl.Float64) * k,
        )
        for k in range(1, cellcount + 1)
    ])

    def get_linebinned(dfestimators: pl.DataFrame) -> npt.NDArray[np.float64]:
        dfopacities, _ = at.lines.plotopacity.get_massweighted_opacities(
            adata, time_days, dfestimators, at.lines.core.get_lambda_bin_edges(3000.0, 4000.0, 10.0)
        )
        return dfopacities["linebinned"].to_numpy()

    linebinned_onecell = get_linebinned(dfcell)
    assert linebinned_onecell.max() > 0.0
    meanfactor = (2 * cellcount + 1) / 3
    assert np.allclose(get_linebinned(dfcells), meanfactor * linebinned_onecell, rtol=1e-10, atol=0.0)


def test_plotopacity_calculates_only_the_bins_of_the_plot(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The bins of the plot range give the same plotted data as a calculation of the full grid of rpkt.h.

    The range also holds one bin more than half the window of the moving average at each end. The width of a bin
    comes from rpkt.h, and the command prints the range and the number of the bins.
    """
    (tmp_path / "artis").mkdir()
    (tmp_path / "artis" / "rpkt.h").write_text(
        "constexpr double expopac_lambdamin = 3000.;\n"
        "constexpr double expopac_lambdamax = 4000.;\n"
        "constexpr double expopac_deltalambda = 10.;\n",
        encoding="utf-8",
    )
    assert at.lines.core.get_expopac_grid(tmp_path) == (3000.0, 4000.0, 10.0)
    assert at.lines.core.get_expopac_grid(modelpath) is None

    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    adata = at.lines.core.get_opacity_atomic_data(modelpath)
    dfcell = at.lines.core.get_cell_estimators(modelpath, timestep, None, "Te")
    dffull, _ = at.lines.plotopacity.get_massweighted_opacities(
        adata, time_days, dfcell, at.lines.core.get_lambda_bin_edges(3000.0, 4000.0, 10.0)
    )
    xmin, xmax, width = 3213.0, 3517.0, 50.0
    capsys.readouterr()
    edges, deltalambda = at.lines.plotopacity.get_computed_bin_edges(tmp_path, xmin, xmax, None, width)
    assert deltalambda == pytest.approx(10.0, rel=1e-12, abs=0.0)
    assert len(edges) - 1 < 50 < dffull.height
    assert (
        f"{len(edges) - 1} wavelength bins of 10 Angstroms from {edges[0]:g} to {edges[-1]:g} Angstroms of the 100 bins"
        " of rpkt.h from 3000 to 4000 Angstroms"
    ) in capsys.readouterr().out
    dfpart, _ = at.lines.plotopacity.get_massweighted_opacities(adata, time_days, dfcell, edges)

    windowbins = at.lines.plotopacity.get_window_bins(width, deltalambda)
    plotted = [
        (
            df.filter(pl.col("lambda_angstroms_upper") > xmin, pl.col("lambda_angstroms_lower") < xmax),
            at.misc.df_filter_minmax_bracketed(
                at.lines.plotopacity.get_moving_averages(df, windowbins, ["exopac", "linebinned", "linebinned_cap1"]),
                "lambda_angstroms_bin_mid",
                xmin,
                xmax,
            ).collect(),
        )
        for df in (dffull, dfpart)
    ]
    pltest.assert_frame_equal(plotted[0][0], plotted[1][0], rel_tol=1e-9)
    pltest.assert_frame_equal(plotted[0][1], plotted[1][1], rel_tol=1e-9)


def test_plotopacity_window_holds_the_nearest_odd_number_of_bins() -> None:
    """The window of the moving average holds the odd number of bins that is nearest to its width.

    2 * round(n / 2) + 1 gave the odd number nearest to n + 1, e.g. 5 bins for a width of 3 bins.
    """
    windowbins = {
        width: at.lines.plotopacity.get_window_bins(width, 20.0) for width in (20.0, 60.0, 62.0, 140.0, 220.0)
    }
    assert windowbins == {20.0: 1, 60.0: 3, 62.0: 3, 140.0: 7, 220.0: 11}
    # a width of an even number of bins is one bin from two odd numbers, and it takes the larger one
    assert at.lines.plotopacity.get_window_bins(200.0, 20.0) == 11
    assert at.lines.plotopacity.get_window_bins(1.2, 0.2) == 7


def test_opacity_cell_batches_hold_fewer_cells_for_more_bins() -> None:
    """A batch has one row for each cell and bin, thus a batch of more bins must hold fewer cells.

    Each batch held 4096 cells, and the 4998 bins of the ejectaopacity defaults then took 3.6 GB for one batch.
    """
    dfcells = pl.DataFrame({"modelgridindex": range(10000)})
    for numbins in (100, 1200, 4998, 49980):
        batches = at.lines.core.get_cell_batches(dfcells, numbins)
        assert sum(batch.height for batch in batches) == dfcells.height
        assert max(batch.height for batch in batches) * numbins <= at.lines.core.ROWSPERBATCH, numbins


def test_plotopacity_median_of_the_cells_agrees_with_the_median_of_each_bin() -> None:
    """-cellaverage median takes the mass-weighted median of the cells from a histogram of log10 of the opacity.

    The median of each wavelength bin agrees with the exact mass-weighted median within half a step of the histogram.
    A NaN opacity stopped the median with an error of the strict cast of its bin. It now gives NaN, as in the mean.
    """
    from artistools.lines.core import get_expansion_opacities
    from artistools.lines.core import get_opacity_lines

    timestep = 25
    time_days = at.get_timestep_times(modelpath_classic_3d, loc="mid")[timestep]
    dfestimators = at.lines.core.get_cell_estimators(modelpath_classic_3d, timestep, None, "TJ").head(60)
    adata = at.lines.core.get_opacity_atomic_data(modelpath_classic_3d)
    edges = [4000.0 + 100.0 * index for index in range(11)]
    dfmedian, _ = at.lines.plotopacity.get_massweighted_opacities(
        adata, time_days, dfestimators, edges, taucaps=[], cellaverage="median"
    )
    dfcells = get_expansion_opacities(
        get_opacity_lines(adata, dfestimators.columns, edges, time_days), dfestimators, edges, time_days, []
    )
    dfexact = (
        dfcells
        .group_by("lambda_angstroms_binindex")
        .agg(exact=at.misc.get_weighted_quantile_expr("exopac", "mass_g", 0.5))
        .sort("lambda_angstroms_binindex")
    )
    ratio = dfmedian["exopac"].to_numpy() / dfexact["exact"].to_numpy()
    assert np.all(np.abs(np.log10(ratio)) <= 0.5 * at.lines.plotopacity.OPACITYMEDIAN_LOGSTEP + 1e-9)

    dfnancell = dfestimators.with_columns(
        T_exc=pl
        .when(pl.col("modelgridindex") == dfestimators["modelgridindex"][0])
        .then(math.nan)
        .otherwise(pl.col("T_exc"))
    )
    for cellaverage in ("mean", "median"):
        dfnan, _ = at.lines.plotopacity.get_massweighted_opacities(
            adata, time_days, dfnancell, edges, taucaps=[], cellaverage=cellaverage
        )
        assert dfnan["exopac"].is_nan().all(), cellaverage


def test_plotopacity_velocity_range_takes_the_cells_of_the_range(capsys: pytest.CaptureFixture[str]) -> None:
    """-vmin and -vmax of plotopacity select the cells with a mid-point velocity in the range, and no other cell.

    Each velocity needs a unit. The count of the cells in the range and the title give each velocity in the unit
    of the user.
    """
    lzmodel, modelmeta = at.scan_modeldata(modelpath_classic_3d)
    dfvelocities = (
        at.inputmodel
        .add_derived_cols_to_modeldata(lzmodel, modelmeta=modelmeta)
        .filter(pl.col("rho") > 0.0)
        .select("modelgridindex", "vel_r_mid_on_c", costheta=pl.col("vel_z_mid_on_c") / pl.col("vel_r_mid_on_c"))
        .collect()
    )
    expectedcells = set(dfvelocities.filter(pl.col("vel_r_mid_on_c").is_between(0.1, 0.2))["modelgridindex"])
    assert 0 < len(expectedcells) < dfvelocities.height

    args = at.misc.parse_cli_args(at.lines.plotopacity.addargs, None, None, ["-vmin", "0.1c", "-vmax", "0.2c"])
    dfestimators = at.lines.core.get_cell_estimators(modelpath_classic_3d, 5, None, "Te")
    capsys.readouterr()
    dfcells = at.lines.plotopacity.select_cells_in_ranges(dfestimators, args.vmin, args.vmax)
    assert set(dfcells["modelgridindex"]) == expectedcells
    assert (
        f"{len(expectedcells)} of {dfestimators.height} cells with estimators are in the range"
        " with vmin = 0.1c and vmax = 0.2c" in (capsys.readouterr().out)
    )

    # -costhetamin selects the cells near the pole at +z, and -dirbin 90 selects the same cells
    expectedpolar = set(
        dfvelocities.filter((pl.col("vel_r_mid_on_c") >= 0.1) & (pl.col("costheta") >= 0.8))["modelgridindex"]
    )
    assert 0 < len(expectedpolar) < len(set(dfestimators["modelgridindex"]))
    for selectionargs in (["-costhetamin", "0.8"], ["-dirbin", "90"]):
        args = at.misc.parse_cli_args(at.lines.plotopacity.addargs, None, None, ["-vmin", "0.1c", *selectionargs])
        costhetarange = at.misc.get_costheta_range(args)
        dfpolarcells = at.lines.plotopacity.select_cells_in_ranges(dfestimators, args.vmin, args.vmax, costhetarange)
        assert set(dfpolarcells["modelgridindex"]) == expectedpolar & set(dfestimators["modelgridindex"])

    args = at.misc.parse_cli_args(at.lines.plotopacity.addargs, None, None, ["-vmin", "0.1c", "-vmax", "60000km/s"])
    assert at.lines.plotopacity.get_cells_text(None, args.vmin, args.vmax) == (
        "mass-weighted mean of the cells with vmin = 0.1c and vmax = 60000 km/s"
    )
    with pytest.raises(SystemExit):
        at.misc.parse_cli_args(at.lines.plotopacity.addargs, None, None, ["-vmin", "0.1"])

    args = at.misc.parse_cli_args(at.lines.plotopacity.addargs, None, None, ["-vmin", "150000km/s"])
    with pytest.raises(ValueError, match=re.escape("No cell with estimators is in the range with vmin = 150000 km/s")):
        at.lines.plotopacity.select_cells_in_ranges(dfestimators, args.vmin, args.vmax)


def test_cell_estimators_of_an_empty_cell_give_an_error() -> None:
    """A cell with no matter has no estimators, and the error names the cell and the next cell with estimators.

    The command stopped with "cannot concat empty list" before.
    """
    lzmodel, modelmeta = at.scan_modeldata(modelpath_classic_3d)
    emptycell = (
        at.inputmodel
        .add_derived_cols_to_modeldata(lzmodel, modelmeta=modelmeta)
        .filter(pl.col("rho") == 0)
        .select(pl.col("modelgridindex").min())
        .collect()
        .item()
    )
    estimatorcells = at.scan_estimators(modelpath_classic_3d, timestep=5).select("modelgridindex").collect()
    nextcell = estimatorcells.filter(pl.col("modelgridindex") > emptycell)["modelgridindex"].min()
    with pytest.raises(ValueError, match=rf"hold no values for cell {emptycell} .* is cell {nextcell}$"):
        at.lines.core.get_cell_estimators(modelpath_classic_3d, 5, emptycell, "Te")


@mock.patch.object(mplax.Axes, "axhline", side_effect=mplax.Axes.axhline, autospec=True)
@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_plotopacity_writes_the_drawn_values(
    mockplot: mock.MagicMock,
    mockaxhline: mock.MagicMock,
    tmp_path: Path,
    read_data_file: Callable[[Path], pl.DataFrame],
) -> None:
    """--write_data gives the drawn bins, the moving averages, the ratios, the line counts, and the Planck mean.

    The moving average has one more point at each end of the range, thus it goes in a second table. With no moving
    average, the ratios go in the table of the bins, with one value for each bin.
    """
    commonargs = ["-modelpath", str(modelpath), "-timestep", "40", "-xmin", "3000", "-xmax", "4000", "--write_data"]
    at.lines.plotopacity.main(
        argsraw=[*commonargs, "--showplanckmean", "--showlinecount", "-o", str(tmp_path / "opac.pdf")]
    )
    dfbins = read_data_file(tmp_path / "opac.txt")
    dfaverages = read_data_file(tmp_path / "opac_movingaverage.txt")

    axes = list(dict.fromkeys(call.args[0] for call in mockplot.call_args_list))
    mainplots = [call.args for call in mockplot.call_args_list if call.args[0] is axes[0]]
    ratioplots = [call.args for call in mockplot.call_args_list if call.args[0] is axes[1]]
    (linecountplot,) = [call.args for call in mockplot.call_args_list if call.args[0] is axes[2]]
    columns = ["exopac", "linebinned_cap1", "linebinned"]
    # the moving average draws one more point at each end of -xmin and -xmax, and the table takes the points inside
    averagexvalues = np.asarray(mainplots[1][1], dtype=np.float64)
    isinrange = (averagexvalues >= 3000.0) & (averagexvalues <= 4000.0)
    # each opacity draws its bins, then its moving average. The line of each bin has a point at each edge and a NaN
    for column, (_, binx, biny), (_, averagexall, averageyall) in zip(
        columns, mainplots[0::2], mainplots[1::2], strict=True
    ):
        assert np.allclose(dfbins["lambda_angstroms_lower"].to_numpy(), binx[0::3], rtol=1e-12, atol=0.0)
        assert np.allclose(dfbins["lambda_angstroms_upper"].to_numpy(), binx[1::3], rtol=1e-12, atol=0.0)
        assert np.allclose(dfbins[f"{column}_cm2_per_g"].to_numpy(), biny[0::3], rtol=1e-12, atol=0.0)
        averagex, averagey = np.asarray(averagexall)[isinrange], np.asarray(averageyall)[isinrange]
        assert np.allclose(dfaverages["lambda_angstroms_bin_mid"].to_numpy(), averagex, rtol=1e-12, atol=0.0)
        assert np.allclose(dfaverages[f"{column}_movingaverage_cm2_per_g"].to_numpy(), averagey, rtol=1e-12, atol=0.0)
    for column, (_, ratioxall, ratioyall) in zip(columns[1:], ratioplots, strict=True):
        ratiox, ratioy = np.asarray(ratioxall)[isinrange], np.asarray(ratioyall)[isinrange]
        assert np.allclose(dfaverages["lambda_angstroms_bin_mid"].to_numpy(), ratiox, rtol=1e-12, atol=0.0)
        assert np.allclose(
            dfaverages[f"ratio_{column}_to_exopac"].to_numpy(), ratioy, rtol=1e-12, atol=0.0, equal_nan=True
        )
    assert np.array_equal(dfbins["linecount"].to_numpy(), linecountplot[2][0::3])
    assert np.allclose(dfbins["planckmean_exopac_cm2_per_g"], mockaxhline.call_args.args[1], rtol=1e-12, atol=0.0)

    mockplot.reset_mock()
    at.lines.plotopacity.main(argsraw=[*commonargs, "-movingaveragewidth", "0", "-o", str(tmp_path / "noaverage.pdf")])
    assert not (tmp_path / "noaverage_movingaverage.txt").exists()
    dfbins = read_data_file(tmp_path / "noaverage.txt")
    assert "planckmean_exopac_cm2_per_g" not in dfbins.columns
    axes = list(dict.fromkeys(call.args[0] for call in mockplot.call_args_list))
    ratioplots = [call.args for call in mockplot.call_args_list if call.args[0] is axes[1]]
    for column, (_, ratiox, ratioy) in zip(columns[1:], ratioplots, strict=True):
        assert np.allclose(dfbins["lambda_angstroms_lower"].to_numpy(), ratiox[0::3], rtol=1e-12, atol=0.0)
        assert np.allclose(
            dfbins[f"ratio_{column}_to_exopac"].to_numpy(), ratioy[0::3], rtol=1e-12, atol=0.0, equal_nan=True
        )


def test_expansion_opacities_give_zero_for_a_cell_with_no_temperature() -> None:
    """A null T_exc and a T_exc of zero give a zero expansion opacity, and a positive T_exc gives a finite value."""
    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    dfcell = at.lines.core.get_cell_estimators(modelpath, timestep, None, "Te")
    assert dfcell.height == 1
    edges = at.lines.core.get_lambda_bin_edges(3000.0, 4000.0, 20.0)
    lines = at.lines.core.get_opacity_lines(
        at.lines.core.get_opacity_atomic_data(modelpath), dfcell.columns, edges, time_days
    )
    dfcells = pl.concat([dfcell, dfcell, dfcell]).with_columns(
        modelgridindex=pl.Series([0, 1, 2], dtype=dfcell["modelgridindex"].dtype),
        T_exc=pl.Series([None, 0.0, dfcell["T_exc"].item()], dtype=pl.Float64),
    )
    dfbins = at.lines.core.get_expansion_opacities(lines, dfcells, edges, time_days)
    for column in at.lines.core.get_opacity_columns(at.lines.core.DEFAULT_TAUCAPS):
        values = dfbins.group_by("modelgridindex", maintain_order=True).agg(pl.col(column).abs().sum())[column]
        assert values[0] == pytest.approx(0.0)
        assert values[1] == pytest.approx(0.0)
        assert np.isfinite(values[2])
        assert values[2] > 0.0


def test_opacity_levels_are_the_levels_that_artis_keeps() -> None:
    """The opacities take only the first nlevelsmax levels of each ion from compositiondata.txt, as ARTIS does.

    The partition function summed all 1792 levels of Fe I in adata.txt. ARTIS keeps 500 of them in the classic test
    model, and the sum of all the levels is 1.97 times the sum of the kept levels at 20000 K.
    """
    classicmodelpath = at.get_path("testdata") / "test-classicmode_3d"
    adata = at.lines.core.get_opacity_atomic_data(classicmodelpath)
    keptlevelcount_of_element = at.atomic.get_kept_level_counts(classicmodelpath)
    alllevels = at.atomic.get_levels(classicmodelpath)
    for Z, ion_stage, dflevels in adata.select("Z", "ion_stage", "levels").iter_rows():
        levelcount = alllevels.filter(pl.col("Z") == Z, pl.col("ion_stage") == ion_stage)["levels"].item().height
        keptlevelcount = keptlevelcount_of_element[Z]
        assert dflevels.height == (levelcount if keptlevelcount is None else min(keptlevelcount, levelcount))
    assert adata.filter(pl.col("Z") == 26, pl.col("ion_stage") == 1)["levels"].item().height == 500

    # the test model keeps all the levels, because its compositiondata.txt gives -1
    assert at.get_composition_data(modelpath)["nlevelsmax_readin"].to_list() == [-1, -1]
    testmodeladata = at.lines.core.get_opacity_atomic_data(modelpath)
    pltest.assert_series_equal(
        testmodeladata.select(pl.col("levels").map_elements(len, return_dtype=pl.Int64)).to_series(),
        at.atomic.get_levels(modelpath).select(pl.col("levels").map_elements(len, return_dtype=pl.Int64)).to_series(),
    )


def test_opacity_lines_name_an_ion_with_no_transition_table() -> None:
    """An ion with no table in transitiondata.txt gives a clear error and not ColumnNotFoundError."""
    adata = at.lines.core.get_opacity_atomic_data(modelpath)
    adata = adata.with_columns(pl.Series("transitions", [pl.LazyFrame() for _ in range(adata.height)], dtype=pl.Object))
    with pytest.raises(ValueError, match=r"transitiondata\.txt holds no table for Fe_II"):
        at.lines.core.get_opacity_lines(adata, ["nnion_Fe_II"], [3000.0, 4000.0], 10.0)
