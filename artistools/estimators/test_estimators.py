import re
import typing as t
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import numpy as np
import polars as pl
import pytest

import artistools as at

modelpath = at.get_path("testdata") / "testmodel"
modelpath_classic_3d = at.get_path("testdata") / "test-classicmode_3d"
outputpath = at.get_path("testoutput")

PLOTLIST_FULL: t.Final = (
    [["initabundances", ["Fe", "Ni_stable", "Ni_56"]]],
    ["nne"],
    ["TR", ["_yscale", "linear"], ["_ymin", 1000], ["_ymax", 22000]],
    ["Te"],
    [["averageionisation", ["Fe", "Ni"]]],
    [["populations", ["Fe I", "Fe II", "Fe III", "Fe IV", "Fe V"]]],
    [["populations", ["Co II", "Co III", "Co IV"]]],
    [["gamma_NT", ["Fe I", "Fe II", "Fe III", "Fe IV"]]],
    ["heating_dep", "heating_coll", "heating_bf", "heating_ff", ["_yscale", "linear"]],
    ["cooling_adiabatic", "cooling_coll", "cooling_fb", "cooling_ff", ["_yscale", "linear"]],
    [(pl.col("heating_coll") - pl.col("cooling_coll")).alias("collisional heating - cooling")],
)

PLOTLIST_IONS: t.Final = (
    [["initabundances", ["Fe", "Ni_stable", "Ni_56"]]],
    ["nne"],
    ["TR", ["_yscale", "linear"], ["_ymin", 1000], ["_ymax", 22000]],
    ["Te"],
    [["averageionisation", ["Fe"]]],
    [["populations", ["Fe I", "Fe II", "Fe III", "Fe IV", "Fe V"]]],
    [["populations", ["Co II", "Co III", "Co IV"]]],
    ["heating_dep", "heating_coll", "heating_bf", "heating_ff", ["_yscale", "linear"]],
    ["cooling_adiabatic", "cooling_coll", "cooling_fb", "cooling_ff", ["_yscale", "linear"]],
)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_estimator_snapshot(mockplot: t.Any) -> None:
    plotlist = PLOTLIST_FULL

    at.estimators.plot(
        argsraw=[],
        modelpath=modelpath,
        plotlist=plotlist,
        outputfile=outputpath / "test_estimator_snapshot",
        timedays=300,
    )
    xarr = [0.0, 4000.0]
    for x in mockplot.call_args_list:
        assert np.allclose(xarr, x[0][1], rtol=1e-3, atol=1e-3)

    # order of keys is important
    expectedvals = {
        "init_fe": 0.10000000149011612,
        "init_nistable": 0.0,
        "init_ni56": 0.8999999761581421,
        "nne": 794211.0,
        "TR": 6932.45,
        "Te": 5776.620000000001,
        "averageionisation_Fe": 1.9453616269532485,
        "averageionisation_Ni": 1.970637712188408,
        "populations_FeI": 4.801001667392128e-05,
        "populations_FeII": 0.350781150587666,
        "populations_FeIII": 0.3951266859004141,
        "populations_FeIV": 0.21184950941623004,
        "populations_FeV": 0.042194644079016,
        "populations_CoII": 0.10471832570699871,
        "populations_CoIII": 0.476333358337709,
        "populations_CoIV": 0.41894831595529214,
        "gamma_NT_FeI": 7.571e-06,
        "gamma_NT_FeII": 3.711e-06,
        "gamma_NT_FeIII": 2.762e-06,
        "gamma_NT_FeIV": 1.702e-06,
        "heating_dep": 6.56117e-10,
        "heating_coll": 2.37823e-09,
        "heating_bf": 1.27067e-13,
        "heating_ff": 1.86474e-16,
        "cooling_adiabatic": 9.72392e-13,
        "cooling_coll": 3.02786e-09,
        "cooling_fb": 4.82714e-12,
        "cooling_ff": 1.62999e-13,
        "collisional heating - cooling": -6.4962990e-10,
    }
    assert len(expectedvals) == len(mockplot.call_args_list)
    yvals = {
        varname: callargs[0][2] for varname, callargs in zip(expectedvals.keys(), mockplot.call_args_list, strict=False)
    }

    print({key: yarr[1] for key, yarr in yvals.items()})

    for varname, expectedval in expectedvals.items():
        assert np.allclose([expectedval, expectedval], yvals[varname], rtol=0.001), (
            varname,
            expectedval,
            yvals[varname][1],
        )


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_estimator_averaging(mockplot: t.Any) -> None:
    plotlist = PLOTLIST_FULL

    at.estimators.plot(
        argsraw=[],
        modelpath=modelpath,
        plotlist=plotlist,
        outputfile=outputpath / "test_estimator_averaging",
        timestep="50-54",
    )

    xarr = [0.0, 4000.0]
    for x in mockplot.call_args_list:
        assert np.allclose(xarr, x[0][1], rtol=1e-3, atol=1e-3)

    # order of keys is important
    expectedvals = {
        "init_fe": 0.10000000149011612,
        "init_nistable": 0.0,
        "init_ni56": 0.8999999761581421,
        "nne": 811131.8125,
        "TR": 6932.65771484375,
        "Te": 5784.4521484375,
        "averageionisation_Fe": 1.9466091928476605,
        "averageionisation_Ni": 1.9673294753348698,
        "populations_FeI": 4.668364835386799e-05,
        "populations_FeII": 0.35026945954378863,
        "populations_FeIII": 0.39508678896764393,
        "populations_FeIV": 0.21220745115264195,
        "populations_FeV": 0.042389615364484115,
        "populations_CoII": 0.1044248111887582,
        "populations_CoIII": 0.4759472294613869,
        "populations_CoIV": 0.419627959349855,
        "gamma_NT_FeI": 7.741022037400234e-06,
        "gamma_NT_FeII": 3.7947153292832773e-06,
        "gamma_NT_FeIII": 2.824587987164586e-06,
        "gamma_NT_FeIV": 1.7406694591346083e-06,
        "heating_dep": 6.849705802558503e-10,
        "heating_coll": 2.4779998053503505e-09,
        "heating_bf": 1.2916119454357833e-13,
        "heating_ff": 2.1250019797070045e-16,
        "cooling_adiabatic": 1.000458830363593e-12,
        "cooling_coll": 3.1562059632506134e-09,
        "cooling_fb": 5.0357105638165756e-12,
        "cooling_ff": 1.7027620090835638e-13,
        "collisional heating - cooling": -6.782059913668093e-10,
    }
    assert len(expectedvals) == len(mockplot.call_args_list)
    yvals = {
        varname: callargs[0][2] for varname, callargs in zip(expectedvals.keys(), mockplot.call_args_list, strict=False)
    }

    print({key: yarr[1] for key, yarr in yvals.items()})

    for varname, expectedval in expectedvals.items():
        assert np.allclose([expectedval, expectedval], yvals[varname], rtol=0.001, equal_nan=True)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_estimator_snapshot_classic_3d(mockplot: t.Any) -> None:
    plotlist = PLOTLIST_IONS

    at.estimators.plot(
        argsraw=[],
        modelpath=modelpath_classic_3d,
        markers=True,
        plotlist=plotlist,
        outputfile=outputpath / "test_estimator_snapshot_classic_3d.pdf",
        timedays=4,
    )

    # order of keys is important
    expected_yvals_mean = {
        "init_fe": 0.015787530690431595,
        "init_nistable": 0.009560450911521912,
        "init_ni56": 0.04967936500906944,
        "nne": 14232720384.0,
        "TR": 19025.818359375,
        "Te": 71311.2109375,
        "averageionisation_Fe": 3.054003953933716,
        "populations_FeI": 5.372131767415029e-16,
        "populations_FeII": 0.0001938836503541097,
        "populations_FeIII": 0.06827619671821594,
        "populations_FeIV": 0.8087993860244751,
        "populations_FeV": 0.12271492183208466,
        "populations_CoII": 0.1702212691307068,
        "populations_CoIII": 0.24963033199310303,
        "populations_CoIV": 0.5801447629928589,
        "heating_dep": 2.5638628358137794e-06,
        "heating_coll": 0.0002122219739248976,
        "heating_bf": 2.178675231334637e-06,
        "heating_ff": 5.598059793499033e-10,
        "cooling_adiabatic": 1.2903782209416903e-10,
        "cooling_coll": 4.360072853160091e-05,
        "cooling_fb": 9.622852559232342e-08,
        "cooling_ff": 6.681727948709693e-10,
    }

    expected_yvals_std = {
        "init_fe": 0.03867174685001373,
        "init_nistable": 0.024418460205197334,
        "init_ni56": 0.13267292082309723,
        "nne": 52205641728.0,
        "TR": 8704.7080078125,
        "Te": 53293.2578125,
        "averageionisation_Fe": 0.3648064434528351,
        "populations_FeI": 1.0559215211458726e-14,
        "populations_FeII": 0.003967594355344772,
        "populations_FeIII": 0.2206096053123474,
        "populations_FeIV": 0.3149721026420593,
        "populations_FeV": 0.25867846608161926,
        "populations_CoII": 0.36867186427116394,
        "populations_CoIII": 0.3848763406276703,
        "populations_CoIV": 0.45789873600006104,
        "heating_dep": 2.4430109988315962e-05,
        "heating_coll": 0.0047865696251392365,
        "heating_bf": 4.846786396228708e-05,
        "heating_ff": 3.555698846469113e-09,
        "cooling_adiabatic": 1.2155411122094506e-09,
        "cooling_coll": 0.0009426323231309652,
        "cooling_fb": 2.1289438336680178e-06,
        "cooling_ff": 7.294685744341223e-09,
    }

    plot_calls_markers = mockplot.call_args_list[1::2]
    assert len(expected_yvals_mean) == len(plot_calls_markers)

    yvals_mean = {
        varname: float(np.array(callargs[0][2]).mean())
        for varname, callargs in zip(expected_yvals_mean.keys(), plot_calls_markers, strict=True)
    }
    print(f"{yvals_mean=}")

    yvals_std = {
        varname: float(np.array(callargs[0][2]).std())
        for varname, callargs in zip(expected_yvals_std.keys(), plot_calls_markers, strict=True)
    }
    print(f"{yvals_std=}")

    for varname, expectedmean in expected_yvals_mean.items():
        assert np.isclose(expectedmean, yvals_mean[varname], rtol=0.01), (varname, expectedmean, yvals_mean[varname])
    for varname, expectedstd in expected_yvals_std.items():
        assert np.isclose(expectedstd, yvals_std[varname], rtol=0.01), (varname, expectedstd, yvals_std[varname])


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_estimator_snapshot_classic_3d_x_axis(mockplot: t.Any) -> None:
    plotlist = PLOTLIST_IONS

    at.estimators.plot(
        argsraw=[],
        modelpath=modelpath_classic_3d,
        plotlist=plotlist,
        outputfile=outputpath / "test_estimator_snapshot_classic_3d_x_axis.pdf",
        timedays=4,
        readonlymgi="alongaxis",
        axis="+x",
    )

    # order of keys is important
    expectedvals = {
        "init_fe": 0.011052947585195368,
        "init_nistable": 0.000944194626933764,
        "init_ni56": 0.002896747941237337,
        "nne": 382033722.1422282,
        "TR": 19732.04,
        "Te": 47127.520000000004,
        "averageionisation_Fe": 3.0271734010069435,
        "populations_FeI": 6.5617829754545176e-24,
        "populations_FeII": 3.161551652102325e-13,
        "populations_FeIII": 0.00010731048012085833,
        "populations_FeIV": 0.9728187853219049,
        "populations_FeV": 0.027125606020167697,
        "populations_CoII": 0.20777361030622207,
        "populations_CoIII": 0.22753057860431092,
        "populations_CoIV": 0.5646079825984672,
        "heating_dep": 5.879422739895874e-08,
        "heating_coll": 0.0,
        "heating_bf": 8.988080000000003e-16,
        "heating_ff": 4.492620000000028e-18,
        "cooling_adiabatic": 1.9406654213040002e-14,
        "cooling_coll": 2.1374800003106965e-14,
        "cooling_fb": 3.376760000131059e-17,
        "cooling_ff": 1.3946640000041897e-17,
    }

    assert len(expectedvals) == len(mockplot.call_args_list)
    yvals = {
        varname: callargs[0][2] for varname, callargs in zip(expectedvals.keys(), mockplot.call_args_list, strict=False)
    }

    print({key: float(np.array(yarr).mean()) for key, yarr in yvals.items()})

    for varname, expectedval in expectedvals.items():
        assert np.allclose(expectedval, np.array(yvals[varname]).mean(), rtol=0.001), (
            varname,
            expectedval,
            yvals[varname][1],
        )


@pytest.mark.benchmark
def test_estimator_timeevolution() -> None:
    at.estimators.plot(
        argsraw=[],
        modelpath=modelpath,
        outputfile=outputpath / "test_estimator_timeevolution",
        plotlist=[["Te", "nne"]],
        modelgridindex=0,
        x="time",
    )


def test_estimparse_xz_high_preset(tmp_path: Path) -> None:
    """An estimator file compressed with xz -9 declares a 64 MiB dictionary and must still be readable."""
    import lzma

    with lzma.open(tmp_path / "estimators_0000.out.xz", "wt", preset=9) as f:
        f.write("timestep 0 modelgridindex 0 TR 2000 Te 3000 W 1 TJ 2000 nne 1.0e5\n")

    dfest = at.rustext.estimparse(tmp_path, 0, 0)
    assert dfest["Te"].to_list() == pytest.approx([3000.0])


def test_estimparse() -> None:
    pldf = at.rustext.estimparse(modelpath, 0, 0)
    assert pldf.height == 100
    assert {"timestep", "modelgridindex", "TR", "Te", "nne", "nnion_Fe_II", "heating_ff"} <= set(pldf.columns)

    firstcell = pldf.sort("timestep", "modelgridindex").row(0, named=True)
    assert firstcell["timestep"] == 0
    assert firstcell["modelgridindex"] == 0
    assert firstcell["nne"] == pytest.approx(71393.3)
    assert firstcell["nnion_Fe_II"] == pytest.approx(80.59)
    # nnelement is the sum over the ion stages of the element
    assert firstcell["nnelement_Fe"] == pytest.approx(6.226e05 + 8.059e01 + 3.940e-24 + 1.586e-27 + 1.010e-27)
    # quantities recorded as X*nne are also stored divided by the electron density
    assert firstcell["Alpha_R_Fe_II"] == pytest.approx(1.821e-07 / 71393.3)


def test_estimparse_missing_file() -> None:
    with pytest.raises(OSError, match="no estimator file found for rank 999"):
        at.rustext.estimparse(modelpath, 999, 999)


@pytest.mark.parametrize(
    ("badline", "errormessage"),
    [
        ("populations    Z=26  1: 6.226e+05  2: NOTANUMBER", 'could not parse "NOTANUMBER" as a number'),
        ("gamma_R        Z=26  12: 1.0", "no roman numeral for ion stage 12"),
        ("gamma_R        Z=999  1: 1.0", "no element symbol for atomic number 999"),
        ("populations    Z=26  1: 1.0  1: 2.0", "a column was given two values for one cell"),
    ],
)
def test_estimparse_malformed_line(tmp_path: Path, badline: str, errormessage: str) -> None:
    """Unparseable estimator data raises a Python exception naming the file and line, never a panic."""
    (tmp_path / "estimators_0000.out").write_text(
        f"timestep 0 modelgridindex 0 TR 2000 Te 2000 W 1 TJ 2000 nne 71393.3\n{badline}\n"
    )

    with pytest.raises(Exception, match=f"estimators_0000.out:2: {re.escape(errormessage)}"):
        at.rustext.estimparse(tmp_path, 0, 0)


def test_add_derived_estimator_columns_preserves_nulls() -> None:
    """Missing nnelement values count as zero, but a null in any other column must not become a real zero."""
    pldf = pl.LazyFrame({
        "timestep": [0, 1],
        "modelgridindex": [0, 0],
        "Te": [5000.0, None],
        "nne": [None, 1.0e8],
        "nnelement_Fe": [2.0, None],
        "nnelement_Ni": [3.0, 4.0],
    })

    dfout = at.estimators.add_derived_estimator_columns(pldf).collect()

    # nnelement nulls are filled with zero and summed into nntot
    assert dfout["nnelement_Fe"].to_list() == [2.0, 0.0]
    assert dfout["nntot"].to_list() == [5.0, 4.0]

    # every other column keeps its nulls
    assert dfout["Te"].to_list() == [5000.0, None]
    assert dfout["nne"].to_list() == [None, 1.0e8]


def test_add_derived_estimator_columns_total_dep() -> None:
    pldf = pl.LazyFrame({
        "timestep": [0],
        "deposition_gamma": [1.0],
        "deposition_positron": [2.0],
        "deposition_alpha": [4.0],
    })

    dfout = at.estimators.add_derived_estimator_columns(pldf).collect()

    assert dfout["total_dep"].to_list() == [7.0]
    assert "nntot" not in dfout.columns


def test_add_derived_estimator_columns_fills_ion_and_isotope_nulls() -> None:
    """A number density column that a whole MPI rank omitted arrives as null, and must be read as zero.

    Within a single estimator file the reader already fills these with zero, so leaving the cross-rank nulls
    alone would give the same missing ion two different values depending on which rank wrote the cell.
    """
    pldf = pl.LazyFrame({
        "timestep": [0, 1],
        "modelgridindex": [0, 1],
        "nnelement_Fe": [2.0, None],
        "nnion_Fe_II": [None, 1.5],
        "nniso_Ni56": [None, 3.0],
        "Te": [5000.0, None],
    })

    dfout = at.estimators.add_derived_estimator_columns(pldf).collect()

    assert dfout["nnion_Fe_II"].to_list() == [0.0, 1.5]
    assert dfout["nniso_Ni56"].to_list() == [0.0, 3.0]
    assert dfout["nnelement_Fe"].to_list() == [2.0, 0.0]

    # temperatures are not number densities, so a null must stay null
    assert dfout["Te"].to_list() == [5000.0, None]


def test_estimparse_rejects_missing_nne(tmp_path: Path) -> None:
    """A '*nne' estimator must not be divided by another cell's electron density.

    The nne column is only as long as the current row once this cell has set it, so a cell header without nne has
    to be an error rather than silently reusing the previous cell's value.
    """
    (tmp_path / "estimators_0000.out").write_text(
        "timestep 0 modelgridindex 0 TR 2000 Te 2000 W 1 TJ 2000 nne 1.0e5\n"
        "Alpha_R*nne    Z=26  2: 2.0e5\n"
        "timestep 0 modelgridindex 1 TR 2000 Te 2000 W 1 TJ 2000\n"
        "Alpha_R*nne    Z=26  2: 4.0e5\n",
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="nne is not set for this cell"):
        at.rustext.estimparse(tmp_path, 0, 0)


def test_estimparse_divides_by_own_cell_nne(tmp_path: Path) -> None:
    """Each cell's '*nne' estimator is divided by that same cell's electron density."""
    (tmp_path / "estimators_0000.out").write_text(
        "timestep 0 modelgridindex 0 TR 2000 Te 2000 W 1 TJ 2000 nne 1.0e5\n"
        "Alpha_R*nne    Z=26  2: 2.0e5\n"
        "timestep 0 modelgridindex 1 TR 2000 Te 2000 W 1 TJ 2000 nne 4.0e5\n"
        "Alpha_R*nne    Z=26  2: 8.0e5\n",
        encoding="utf-8",
    )

    dfest = at.rustext.estimparse(tmp_path, 0, 0).sort("modelgridindex")

    assert dfest["nne"].to_list() == pytest.approx([1.0e5, 4.0e5])
    assert dfest["Alpha_R_Fe_II"].to_list() == pytest.approx([2.0, 2.0])


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_estimator_default_plotlist_skips_absent_elements(mockplot: t.Any) -> None:
    """The built-in plot list names particular elements, which most models do not contain."""
    funcoutpath = outputpath / "test_estimator_default_plotlist"
    funcoutpath.mkdir(exist_ok=True, parents=True)

    # testmodel has no Sr, so the default averageionisation/populations plots must be skipped rather than raising
    at.estimators.plot(argsraw=[], modelpath=modelpath, timedays=300, outputfile=funcoutpath)

    # the element-independent default subplots (rho and TR) are still drawn
    assert len(mockplot.call_args_list) >= 2
    for call in mockplot.call_args_list:
        yvalues = np.array(call[0][2], dtype=float)
        assert np.all(np.isfinite(yvalues))
        assert np.all(yvalues > 0.0)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_estimator_levelpopulation_dn_on_dvel(mockplot: t.Any) -> None:
    """Plotting dN/dv needs the inner shell velocity, which is a derived model column."""
    funcoutpath = outputpath / "test_estimator_levelpopulation_dn_on_dvel"
    funcoutpath.mkdir(exist_ok=True, parents=True)

    at.estimators.plot(
        argsraw=[],
        modelpath=modelpath,
        timedays=300,
        outputfile=funcoutpath,
        plotlist=[[["levelpopulation_dn_on_dvel", ["Fe II 5"]]]],
    )

    assert mockplot.call_args_list, "nothing was plotted"
    yvalues = np.array(mockplot.call_args_list[0][0][2], dtype=float)
    assert np.all(np.isfinite(yvalues))
    assert np.all(yvalues >= 0.0)


def test_get_averageexcitation() -> None:
    """The average excitation energy must be the population-weighted mean level energy of the ion."""
    dfpops = at.nltepops.read_files(modelpath).filter((pl.col("Z") == 26) & (pl.col("ion_stage") == 2))
    timestep = min(dfpops["timestep"].to_list())
    dftexc = pl.LazyFrame({"timestep": [timestep], "modelgridindex": [0], "T_exc": [6000.0]})

    dfavgexc = at.estimators.get_averageexcitation(modelpath, 26, 2, dftexc).collect()
    assert len(dfavgexc) == 1

    avgexc = dfavgexc["averageexcitation"].item()
    ionlevels = (
        at.atomic.get_levels(modelpath).filter((pl.col("Z") == 26) & (pl.col("ion_stage") == 2))["levels"].item()
    )
    assert ionlevels is not None

    # the mean must lie between the lowest and highest level energies of the ion
    dfts = dfpops.filter((pl.col("timestep") == timestep) & (pl.col("modelgridindex") == 0))
    occupiedlevels = dfts.filter(pl.col("level") >= 0)["level"].to_list()
    energiesoccupied = ionlevels.filter(pl.col("levelindex").is_in(occupiedlevels))
    assert energiesoccupied["energy_ev"].min() <= avgexc <= energiesoccupied["energy_ev"].max()

    # weighting by hand over the resolved levels only must bracket the value, which also includes the superlevel
    dfresolved = dfts.filter(pl.col("level") >= 0).join(
        ionlevels.select(level=pl.col("levelindex").cast(pl.Int64), energy_ev="energy_ev"), on="level", how="inner"
    )
    avgexc_resolvedonly = float((dfresolved["energy_ev"] * dfresolved["n_NLTE"]).sum()) / float(dfts["n_NLTE"].sum())
    assert avgexc >= avgexc_resolvedonly, "adding the superlevel population can only raise the mean energy"


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_estimator_averageexcitation_plot(mockplot: t.Any) -> None:
    """The averageexcitation plot item must draw a finite series for each requested ion."""
    funcoutpath = outputpath / "test_estimator_averageexcitation"
    funcoutpath.mkdir(exist_ok=True, parents=True)

    at.estimators.plot(
        argsraw=[],
        modelpath=modelpath,
        plotlist=[[["averageexcitation", ["Fe II", "Fe III"]]]],
        timedays=300,
        outputfile=funcoutpath,
    )

    assert len(mockplot.call_args_list) >= 2
    for call in mockplot.call_args_list:
        yvalues = np.array(call[0][2], dtype=float)
        assert np.all(np.isfinite(yvalues))
        assert np.all(yvalues >= 0.0), "an excitation energy above the ground state cannot be negative"


def test_averageexcitation_plotitem_needs_nlte_files(tmp_path: Path) -> None:
    """The default plot list must drop averageexcitation for a model with no NLTE population files."""
    from artistools.estimators.plotestimators import default_plotitem_has_data

    plotitem = [["averageexcitation", ["Fe II"]]]
    estimatorcolumns = ["timestep", "modelgridindex", "Te", "nnelement_Fe"]

    assert default_plotitem_has_data(plotitem, estimatorcolumns, modelpath)
    assert not default_plotitem_has_data(plotitem, estimatorcolumns, tmp_path)


def test_get_elemcolor_is_stable_and_unbounded() -> None:
    """Every element must have a colour, fixed by the element and not by how many plots came before it.

    Colours used to be handed out in call order from a ten-entry list, so an element's colour depended on the
    plots that preceded it in the same process, and the eleventh element raised IndexError.
    """
    from artistools.estimators.plotestimators import colors_tab10
    from artistools.estimators.plotestimators import elementcolors
    from artistools.estimators.plotestimators import get_elemcolor

    assert get_elemcolor(elsymbol="Fe") == colors_tab10[0]
    assert get_elemcolor(elsymbol="Ni") == colors_tab10[1]
    assert get_elemcolor(elsymbol="Co") == colors_tab10[2]

    # far more elements than the reserved list can hold, in two different orders
    colors_ascending = [get_elemcolor(atomic_number=Z) for Z in range(1, 93)]
    colors_descending = [get_elemcolor(atomic_number=Z) for Z in reversed(range(1, 93))]
    assert colors_ascending == list(reversed(colors_descending))

    assert get_elemcolor(elsymbol="O") == get_elemcolor(atomic_number=8)

    # a reserved colour must stay unique to its element, so that Fe, Ni and Co remain identifiable
    reservedZ = {at.get_atomic_number(elsymbol) for elsymbol in elementcolors}
    unreserved = [get_elemcolor(atomic_number=Z) for Z in range(1, 93) if Z not in reservedZ]
    assert not set(elementcolors.values()).intersection(unreserved)


def test_estimparse_index_columns_are_integers() -> None:
    """The timestep and modelgridindex columns must be parsed as integers, not floats.

    Everything in the cell header used to be stored as f32, which cannot represent every cell number of a large
    3D grid exactly: 26999999 comes back as 27000000.
    """
    dfestim = at.rustext.estimparse(modelpath, 0, 0)

    for colname in ("timestep", "modelgridindex"):
        assert dfestim.schema[colname] == pl.Int32, f"{colname} must stay an exact integer"

    # the physical quantities are still f32
    assert dfestim.schema["Te"] == pl.Float32
