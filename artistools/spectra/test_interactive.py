import argparse
import dataclasses as dc
import math
import shlex
import sys
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import matplotlib.colors as mplcolors
import matplotlib.figure as mplfig
import matplotlib.patches as mpatches
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

import artistools as at
from artistools.spectra import interactive
from artistools.spectra import plotspectra
from artistools.viewertools import application as viewerapplication
from artistools.viewertools import core as viewercore
from artistools.viewertools import series as viewerseries
from artistools.viewertools.window import apply_values
from artistools.viewertools.window import draw_values
from artistools.viewertools.window import fit_one_frame_column

modelpath = at.get_path("testdata") / "testmodel"
modelpath_classic_3d = at.get_path("testdata") / "test-classicmode_3d"


def test_deltalogx_largestscale_is_the_diagonal_of_a_cell() -> None:
    """-deltalogx largestscale gives Δv/c of the diagonal of a cell of a 3D grid, and the viewer keeps the keyword."""
    vmax_cmps = 2892020000.0
    expected = math.sqrt(3) * 2 * vmax_cmps / 10 / at.constants.C_cm_per_s
    args = argparse.Namespace(deltalogx="largestscale", modelspecpaths=[modelpath_classic_3d])
    plotspectra.resolve_deltalogx_scale(args)
    assert math.isclose(args.deltalogx, expected, rel_tol=1e-6)

    viewer = make_headless_viewer([str(modelpath_classic_3d), "-t", "5", "-deltalogx", "largestscale", "--interactive"])
    assert viewer.values.deltalogx == "largestscale"
    assert "-deltalogx largestscale" in viewer.get_command()
    assert viewer.drawndeltalogx is not None
    assert math.isclose(viewer.drawndeltalogx, expected, rel_tol=1e-6)


def test_explicit_linear_yscale_with_logscaley_stops_the_command() -> None:
    """An explicit -yscale linear gives a different scale from --logscaley, also when linear is the default.

    resolve_yscale compared -yscale with the default of the command, thus it did not see an explicit linear as a
    choice of the user.
    """
    with pytest.raises(SystemExit):
        at.misc.parse_cli_args(plotspectra.addargs, None, None, [str(modelpath), "-yscale", "linear", "--logscaley"])
    # a keyword becomes a default of the parser, and it is still a choice of the user
    with pytest.raises(SystemExit):
        at.misc.parse_cli_args(plotspectra.addargs, None, None, None, {"yscale": "linear", "logscaley": True})

    # the viewer has no box item for the alias "lin", thus the resolved scale has the spelling "linear"
    args = at.misc.parse_cli_args(plotspectra.addargs, None, None, [str(modelpath), "-yscale", "lin"])
    assert (args.yscale, args.logscaley) == ("linear", False)
    viewer = make_headless_viewer([str(modelpath), "-t", "300", "-yscale", "lin", "--interactive"])
    assert viewer.values.yscale in viewercore.get_yscale_choices(viewer.parser)


def test_interactive_command_tokens() -> None:
    """The command of the viewer drops each form of an option that a control sets, and keeps the other options."""
    parser = viewercore.make_parser(plotspectra.addargs)
    tokens = [
        "my model",
        "sn2011fe_PTF11kly_20120822_norm.txt",
        "-t300",
        "-xmin=3000",
        "-lambdamax",
        "9000",
        "-label",
        "foo",
        "bar",
        "--interactive",
        "--notimeclamp",
        "-timemin",
        "290",
        "-timema",
        "320",
        "-ts70",
        "-time",
        "300",
        "--emissionabsorption",
        "-maxseriescount",
        "20",
        "-groupby",
        "nuc",
        "-deltax=20",
        "--nostack",
        "-fixedionlist",
        "Fe II",
        "Co II",
        "-plotviewingangle",
        "-1",
        "--",
        "-folder",
    ]
    basetokens = viewercore.remove_options(parser, tokens, interactive.CONTROLLED_DESTS | viewercore.WINDOW_DESTS)
    assert viewercore.make_command_tokens(basetokens, ["-t", "306", "-xmin", "3000", "-xmax", "9000"]) == [
        "my model",
        "sn2011fe_PTF11kly_20120822_norm.txt",
        *("-t", "306", "-xmin", "3000", "-xmax", "9000"),
        # the viewer removes -plotviewingangle and also its negative value, which starts with "-"
        *("-label", "foo", "bar", "--", "-folder"),
    ]


def test_interactive_time_range_argument() -> None:
    """A snapped -timedays value must select its timesteps, and a continuous range must stay inside the valid times."""
    tmids, tstarts, tends = (at.get_timestep_times(modelpath, loc=loc) for loc in ("mid", "start", "end"))
    pairs = [(54, 58), (0, 3), (96, 99), (10, 11), *((timestep, timestep) for timestep in range(len(tmids)))]
    for first, last in pairs:
        timedays = interactive.get_snapped_timedays_argument(tmids, tstarts, tends, first, last)
        assert at.misc.get_time_range(modelpath, timedays_range_str=timedays)[:2] == (first, last), timedays
    assert interactive.get_snapped_timedays_argument(tmids, tstarts, tends, 54, 54) == "300"

    assert interactive.get_timedays_argument(306.4, 0.0, (250.0, 350.0)) == "306.4"
    assert interactive.get_timedays_argument(306.4, 5.0, (250.0, 350.0)) == "303.9-308.9"
    # the valid times of the test model start at 256.67 d, thus the range starts there
    assert interactive.get_timedays_argument(260.0, 10.0, (256.67, 333.82)) == "256.67-265"

    # format_days rounded a time at a bound to 6 decimals, and the time went outside the valid times. plotspectra
    # rejects such a time
    bounds = (0.1413524, 79.9999996)
    for centre, width in ((0.1413, 0.0), (0.1414, 0.001), (79.99999, 0.1), (40.0, 100.0)):
        for text in interactive.get_timedays_argument(centre, width, bounds).split("-"):
            assert bounds[0] <= float(text) <= bounds[1], (centre, width, text)


def test_interactive_continuous_range_keeps_its_bounds() -> None:
    """A --notimeclamp range of days keeps its bounds when the viewer opens, also inside one timestep.

    The viewer changed a range that holds the middle of one timestep or of no timestep into a single time, and a
    single time reads the whole timestep.
    """
    viewer = make_headless_viewer([str(modelpath), "-t", "299.5-301", "--notimeclamp", "--interactive"])
    tokens = shlex.split(viewer.get_command())
    assert tokens[tokens.index("-t") + 1] == "299.5-301"

    # a continuous single time takes the width of Δ ln t, because a width of 0 reads the whole timestep
    viewer = make_headless_viewer([str(modelpath), "-t", "300", "--notimeclamp", "--interactive"])
    assert viewer.values.widthmode == "dlogt"
    assert viewer.values.width > 0.0


def test_interactive_continuous_width_is_never_zero() -> None:
    """Each width mode of a continuous range gives a width above 0, and the width follows the time.

    A continuous width of 0 selected the whole timestep that holds the time, which is the clamped range.
    """
    viewer = make_headless_viewer([str(modelpath), "-t", "299-301", "--notimeclamp", "--interactive"])
    assert (viewer.values.widthmode, viewer.values.width) == ("days", 2.0)
    # a width of 0 in days changes to Δ ln t, which starts with the Δ ln t of each timestep of the logarithmic grid
    assert apply_values(viewer, viewer.clamp_time(dc.replace(viewer.values, width=0.0))) is None
    assert viewer.values.widthmode == "dlogt"
    assert viewer.values.dlogt == pytest.approx(math.log(viewer.grid.tends[0] / viewer.grid.tstarts[0]), rel=1e-3)
    # the width follows the time
    assert (
        apply_values(viewer, viewer.clamp_time(dc.replace(viewer.values, widthmode="dlogt", dlogt=0.01, centre=290.0)))
        is None
    )
    low, high = (viewer.values.centre + sign * viewer.values.width / 2.0 for sign in (-1.0, 1.0))
    assert math.log(high / low) == pytest.approx(0.01, rel=1e-3)
    # the Down key never gives a width of 0 or less, and it gives a width in days
    assert apply_values(viewer, dc.replace(viewer.values, widthmode="days", width=0.001)) is None
    assert viewer.step_width(-1).width == 0.001
    assert viewer.step_width(1).widthmode == "days"


def test_interactive_valid_timesteps() -> None:
    """The time controls stay inside the valid times, thus a step after the last valid timestep gives None."""
    viewer = make_headless_viewer([str(modelpath), "--interactive"])
    validstart, validend = viewer.grid.timebounds
    assert viewer.grid.tstarts[viewer.grid.validtimesteps[0]] >= validstart > viewer.grid.tstarts[0]
    assert viewer.grid.tends[viewer.grid.validtimesteps[-1]] <= validend < viewer.grid.tends[-1]
    viewer.values = viewer.move_to_end(last=True)
    lasttimestep = viewer.grid.validtimesteps[-1]
    assert interactive.get_grid_selection(viewer.grid, viewer.values) == (lasttimestep, lasttimestep)
    assert viewer.step_time(1) is None
    assert draw_values(viewer) is None

    # a continuous range at each end of the valid times keeps its Δ ln t, and plotspectra accepts it
    viewer = make_headless_viewer([str(modelpath), "-t", "300", "--notimeclamp", "--interactive"])
    # each end starts from the values of the command, thus it does not keep the width mode of the other end
    base = viewer.values
    for bound in (validstart, validend):
        assert apply_values(viewer, viewer.clamp_time(dc.replace(base, centre=bound))) is None
        assert viewer.values.centre == bound
        low, high = (bound + sign * viewer.values.width / 2.0 for sign in (-1.0, 1.0))
        assert math.log(high / low) == pytest.approx(viewer.values.dlogt, rel=1e-3)
        assert apply_values(viewer, dc.replace(viewer.values, centre=bound, widthmode="days", width=1.0)) is None


def test_interactive_up_key_at_the_last_timestep_widens_the_range_backwards() -> None:
    """The Up key at the last valid timestep adds the timestep before the range, as the estimators viewer does.

    The viewer kept the first timestep of the range and clamped the last one, thus the Up key did nothing there.
    """
    viewer = make_headless_viewer([str(modelpath), "--interactive"])
    validtimesteps = viewer.grid.validtimesteps
    viewer.values = viewer.move_to_end(last=True)
    wider = viewer.step_width(1)
    assert interactive.get_grid_selection(viewer.grid, wider) == (validtimesteps[-2], validtimesteps[-1])


def make_headless_viewer(tokens: list[str]) -> interactive.SpectrumViewer:
    """Return a viewer that draws on a canvas with no window, after its first plot."""
    fig = mplfig.Figure()
    FigureCanvasAgg(fig)
    viewer = interactive.SpectrumViewer(tokens, fig)
    assert draw_values(viewer) is None
    return viewer


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_interactive_command_reproduces_plot(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """The command that the viewer shows must draw the same data as the viewer."""
    viewer = make_headless_viewer([str(modelpath), "-t", "290", "--interactive"])
    # a continuous range gives --notimeclamp, and the snapped range below gives whole timesteps
    continuous = dc.replace(
        viewer.values, notimeclamp=True, widthmode="days", centre=306.4, width=5.0, xmin="3000", xmax="9000"
    )
    assert apply_values(viewer, continuous) is None
    assert viewer.get_command().endswith(" -t 303.9-308.9 --xmin 3000 --xmax 9000 --notimeclamp")
    # a continuous range reads the packets that arrive inside it, and not whole timesteps
    assert viewer.get_time_range_text() == "Packets from 303.90 to 308.90 d (Δt = 5.00 d)"
    assert apply_values(viewer, viewer.snap(viewer.values, 58, 62)) is None
    assert "Timesteps 58 to 62" in viewer.get_time_range_text()
    command = viewer.get_command()

    # each plot clears the frame first, thus the frame holds only the series of the model
    [viewerline] = viewer.axes[0].get_lines()
    viewerx, viewery = np.asarray(viewerline.get_xdata()), np.asarray(viewerline.get_ydata())

    mockplot.reset_mock()
    at.spectra.plot(argsraw=[*shlex.split(command)[2:], "-o", str(tmp_path / "spectrum.pdf")])
    assert mockplot.call_count == 1
    assert np.allclose(np.array(mockplot.call_args[0][1]), viewerx, rtol=1e-12, atol=0.0)
    assert np.allclose(np.array(mockplot.call_args[0][2]), viewery, rtol=1e-12, atol=0.0)


def test_interactive_emission_options() -> None:
    """A change that plotspectra rejects must keep the old values, and --showabsorption draws a taller frame."""
    # the test model has no emission.out, thus plotspectra cannot draw its emission plot
    viewer = make_headless_viewer([str(modelpath), "-t", "300", "--interactive"])
    oldvalues, command = viewer.values, viewer.get_command()
    # the status line gives the first line of the error, which names the missing files
    message = apply_values(viewer, dc.replace(viewer.values, showemission=True))
    assert message is not None
    assert message.startswith("None of these files exist")
    assert viewer.values == oldvalues
    assert viewer.get_command() == command

    # the viewer stores the default grouping as None. A stored "ion" was not equal to the None of the controls, thus a
    # change of an emission option removed the series lock
    viewer = make_headless_viewer([str(modelpath_classic_3d), "-t", "5", "--groupby", "ion", "--interactive"])
    assert viewer.values.groupby is None
    figheight = viewer.fig.get_figheight()
    assert apply_values(viewer, dc.replace(viewer.values, showabsorption=True, maxseriescount=5, nostack=True)) is None
    assert viewer.get_command().endswith(" --showabsorption --maxseriescount 5 --nostack")
    assert viewer.fig.get_figheight() > figheight

    # the window disables a choice that plotspectra rejects, thus the test must find the rejection without a plot
    rejection = viewer.get_rejection(dc.replace(viewer.values, groupby="nuc"))
    assert rejection is not None
    assert "nuclide" in rejection


def test_interactive_xunit_and_references() -> None:
    """A new x unit converts and sorts the limits, and a reference spectrum goes after the model paths."""
    viewer = make_headless_viewer([str(modelpath), "-t", "300", "--deltax", "20", "--interactive"])
    # a frequency increases where the wavelength decreases, thus the limit of 19000 Å becomes the minimum
    inhertz = interactive.convert_xunit(viewer.values, "hz", gamma=False)
    assert (inhertz.xmin, inhertz.xmax, inhertz.deltax) == ("1.578e+14", "1.199e+15", "")
    back = interactive.convert_xunit(inhertz, "angstrom", gamma=False)
    assert np.isclose(float(back.xmin), 2500.0, rtol=1e-3, atol=0.0)
    assert np.isclose(float(back.xmax), 19000.0, rtol=1e-3, atol=0.0)

    reference = "sn2011fe_PTF11kly_20120822_norm.txt"
    assert apply_values(viewer, dc.replace(inhertz, spectra=(*inhertz.spectra, reference))) is None
    tokens = shlex.split(viewer.get_command())[2:]
    assert tokens[:4] == [str(modelpath), reference, "-t", "300"]
    assert "--xunit" in tokens
    assert len(viewer.axes[0].get_lines()) == 2


def test_interactive_spectra_hold_the_models_and_the_references(monkeypatch: pytest.MonkeyPatch) -> None:
    """The list of spectra holds each ARTIS model and each reference, also the model of the working folder.

    A command with no path reads the model of the working folder, thus the list shows that model first and the
    command still gives no path. A model that the user adds reaches the command and the plot.
    """
    monkeypatch.chdir(modelpath)
    viewer = make_headless_viewer(["-t", "300", "--interactive"])
    assert viewer.values.spectra == (".",)
    assert viewerseries.get_series_item_text(interactive.REFERENCE_DATA, ".") == f"Model: {modelpath.absolute()}"
    assert shlex.split(viewer.get_command())[2:4] == ["-t", "300"]

    reference = "sn2011fe_PTF11kly_20120822_norm.txt"
    # the second model is the same model folder, because the other test models do not cover 300 days
    spectra = (*viewer.values.spectra, str(modelpath), reference)
    assert apply_values(viewer, dc.replace(viewer.values, spectra=spectra)) is None
    assert shlex.split(viewer.get_command())[2:5] == [".", str(modelpath), reference]
    assert len(viewer.axes[0].get_lines()) == 3
    assert viewerseries.get_series_item_text(interactive.REFERENCE_DATA, reference).startswith("Reference: /")


def test_interactive_new_first_model_gives_the_time_grid() -> None:
    """A new order of the models gives the time controls the timesteps of the new first model.

    A snapped range keeps its middle and its count of timesteps on the new grid. The new first model has longer
    timesteps, thus the days of the old range contain only 3 of them.
    """
    classic1dpath = at.get_path("testdata") / "test-classicmode_1d"
    viewer = make_headless_viewer([str(modelpath_classic_3d), str(classic1dpath), "-t", "4.5-5", "--interactive"])
    first, last = interactive.get_grid_selection(viewer.grid, viewer.values)
    assert last - first + 1 == 4
    centre = viewer.values.centre

    viewer.values = interactive.set_models(viewer, [str(classic1dpath), str(modelpath_classic_3d)], "")
    assert viewer.grid.modelfolders[0] == classic1dpath
    first, last = interactive.get_grid_selection(viewer.grid, viewer.values)
    assert last - first + 1 == 4
    assert viewer.values.centre == pytest.approx(centre, rel=0.05)
    assert viewer.grid.tmids[first] < centre < viewer.grid.tmids[last]


def test_interactive_time_grid_of_a_later_model() -> None:
    """A model that is not first can give its timesteps to the time controls, and the valid times apply to all models.

    plotspectra rejects a time outside the timesteps of each model, thus the valid times stay inside both models.
    """
    classic1dpath = at.get_path("testdata") / "test-classicmode_1d"
    spectra = [str(modelpath_classic_3d), str(classic1dpath)]
    viewer = make_headless_viewer([*spectra, "-t", "4.5-5", "--interactive"])
    viewer.values = interactive.set_models(viewer, spectra, str(classic1dpath))
    assert viewer.grid.gridfolder == classic1dpath
    assert viewer.values.spectra == tuple(spectra)
    first, last = interactive.get_grid_selection(viewer.grid, viewer.values)
    assert last - first + 1 == 4
    for modeltimes in viewer.grid.modeltimes.values():
        assert modeltimes.tstart <= viewer.grid.timebounds[0] < viewer.grid.timebounds[1] <= modeltimes.tend

    # after the removal of the model, the time controls use the timesteps of the first model again
    viewer.values = interactive.set_models(viewer, spectra[:1], viewer.values.timegrid)
    assert (viewer.values.timegrid, viewer.grid.gridfolder) == ("", modelpath_classic_3d)


def test_interactive_time_fits_the_timestep_of_each_model() -> None:
    """A time inside the valid times can still give a timestep of a different model that ends outside its valid times.

    plotspectra clamps the time to the timestep of each model, and it then rejects the days of that timestep.
    """
    classic1dpath = at.get_path("testdata") / "test-classicmode_1d"
    tstarts = at.get_timestep_times(classic1dpath, loc="start")
    tends = at.get_timestep_times(classic1dpath, loc="end")
    timestep = 25
    # the valid times end in the middle of the timestep
    validend = (tstarts[timestep] + tends[timestep]) / 2.0
    modeltimes = interactive.ModelTimes(tstart=tstarts[0], tend=tends[-1], validstart=tstarts[0], validend=validend)
    earlytime = f"{tstarts[timestep] + 0.25 * (tends[timestep] - tstarts[timestep]):.6f}"
    assert float(earlytime) < validend
    assert not interactive.fits_each_model([(classic1dpath, modeltimes)], earlytime)
    assert interactive.fits_each_model([(classic1dpath, modeltimes)], f"{tstarts[timestep - 1] + 0.001:.6f}")


def test_interactive_series_styles_stay_on_their_spectrum() -> None:
    """A new order of the spectra keeps each --label and each --color on its spectrum.

    Each option gives its values in the order of the spectra. Thus a spectrum with no value in front of a spectrum with
    a value takes the token default, and plotspectra then gives it the automatic label, e.g. with the time.
    """
    reference = "sn2011fe_PTF11kly_20120822_norm.txt"
    oldspectra = (str(modelpath), reference, str(modelpath_classic_3d))
    rows = (("--label", ("First",)), ("--color", ("red", "blue")))
    newspectra = (str(modelpath_classic_3d), reference, str(modelpath))

    moved = viewercore.move_series_styles(rows, oldspectra, newspectra)
    assert dict(moved) == {"--label": ("default", "default", "First"), "--color": ("default", "blue", "red")}
    # the values of a removed spectrum go out of the option rows
    assert dict(viewercore.move_series_styles(rows, oldspectra, oldspectra[1:])) == {"--color": ("blue",)}

    viewer = make_headless_viewer([*oldspectra[:2], "-t", "300", "--interactive"])
    labelledrows = viewercore.set_series_rows(
        viewer.values.otheroptions, oldspectra[:2], reference, {"--label": "Observed"}
    )
    labelled = dc.replace(viewer.values, otheroptions=labelledrows)
    assert viewercore.get_row_values(labelled.otheroptions, "--label") == ("default", "Observed")
    assert apply_values(viewer, labelled) is None
    legendlabels = [line.get_label() for line in viewer.axes[0].get_lines()]
    assert legendlabels[1] == "Observed"
    assert legendlabels[0].startswith("TEST MODEL +300")
    unlabelledrows = viewercore.set_series_rows(labelledrows, oldspectra[:2], reference, {"--label": None})
    assert viewercore.get_row_values(unlabelledrows, "--label") is None


def test_interactive_list_gives_a_reference_spectrum_its_legend_label(tmp_path: Path) -> None:
    """The list of spectra gives a reference spectrum the label of its line in the legend.

    A reference file with no label in its metadata takes the path of the command as its label. The list showed the
    name of the file, which is not the label in the legend.
    """
    reference = tmp_path / "myref.txt"
    reference.write_bytes(plotspectra.find_reference_spectrum_file("sn2011fe_PTF11kly_20120822_norm.txt").read_bytes())
    viewer = make_headless_viewer([str(modelpath), str(reference), "-t", "300", "--interactive"])
    legendlabel = viewer.axes[0].get_lines()[1].get_label()
    assert legendlabel == str(reference)
    assert viewerseries.get_series_name(interactive.REFERENCE_DATA, str(reference)) == legendlabel


def test_reference_spectrum_names_are_files_that_plotspectra_finds() -> None:
    """Each name that the reference field offers names a spectrum that plotspectra finds by that name.

    The folder holds compressed files and metadata files with no data file, which plotspectra cannot read by
    their own names.
    """
    names = viewerseries.get_reference_names(at.get_path("artistools_dir") / "data" / "refspectra")
    assert "AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57983.969_Phase+1.43d.dat" in names
    assert not any(name.endswith((".meta.yml", ".xz", ".gz", ".zst")) for name in names)
    assert all(plotspectra.find_reference_spectrum_file_or_none(name) is not None for name in names)


def test_interactive_paths_keep_their_place() -> None:
    """Each path of the command stays a path and keeps its place, also a path after an option.

    The viewer took the paths only from the start of the command, and it put the references after the models. Thus a
    path after an option left the command, and -label named the wrong series.
    """
    reference = "sn2011fe_PTF11kly_20120822_norm.txt"
    viewer = make_headless_viewer([
        reference,
        str(modelpath),
        "-label",
        "Observed",
        "Model",
        "-t",
        "300",
        "--interactive",
    ])
    assert shlex.split(viewer.get_command())[2:4] == [reference, str(modelpath)]

    viewer = make_headless_viewer(["-t", "300", "--notitle", str(modelpath), "--interactive"])
    assert viewer.modelpathtokens == [str(modelpath)]
    # the command without the option starts with the path, and the viewer still reads the time of the command
    assert apply_values(viewer, dc.replace(viewer.values, otheroptions=())) is None
    assert "Timestep 54" in viewer.get_time_range_text()

    # -fixedionlist reads each word that follows it, but a folder at the end of the command is a path
    viewer = make_headless_viewer([
        "--showemission",
        "-fixedionlist",
        "Fe II",
        "Co II",
        str(modelpath_classic_3d),
        "--interactive",
    ])
    assert viewer.modelpathtokens == [str(modelpath_classic_3d)]
    assert str(modelpath_classic_3d) in shlex.split(viewer.get_command())


def test_interactive_command_gives_the_series_count_of_the_box() -> None:
    """The command gives the count of the box, also when that count is the default without -fixedionlist.

    plotspectra gives a missing count the length of -fixedionlist, thus a count of 14 with a list needs the option.
    """
    viewer = make_headless_viewer([str(modelpath_classic_3d), "-t", "4", "--showemission", "--interactive"])
    series = viewer.get_drawn_series()
    for fixedionlist, maxseriescount in ((series[:3], plotspectra.DEFAULT_MAXSERIESCOUNT), (series[:3], 3), ((), 5)):
        values = dc.replace(viewer.values, fixedionlist=fixedionlist, maxseriescount=maxseriescount)
        args = at.misc.parse_cli_args(plotspectra.addargs, None, None, viewer.get_plot_tokens(values))
        plotspectra.resolve_plot_args(args)
        assert args.maxseriescount == maxseriescount


def test_interactive_fixed_y_axis() -> None:
    """A fixed y axis keeps a negative --ymin through the command, and the limits stay when the time changes."""
    viewer = make_headless_viewer([str(modelpath), "-t", "300", "--interactive"])
    assert apply_values(viewer, dc.replace(viewer.values, ymin="-1e-13", ymax="3e-13")) is None
    # Python 3.13 reads a separate -1e-13 as an option, thus the value joins its flag
    assert " --ymin=-1e-13 --ymax 3e-13" in viewer.get_command()
    assert apply_values(viewer, dc.replace(viewer.values, centre=305.0)) is None
    assert np.allclose(viewer.axes[0].get_ylim(), (-1e-13, 3e-13), rtol=1e-9, atol=0.0)


def test_interactive_figwidthscale_fills_the_plot_area() -> None:
    """The viewer sets --figwidthscale to give the figure the shape of the plot area, and the command shows it."""
    viewer = make_headless_viewer([str(modelpath), "-t", "300", "--figwidthscale", "1.5", "--interactive"])
    for areawidth, areaheight in ((1600.0, 500.0), (600.0, 500.0)):
        figwidthscale = fit_one_frame_column(viewer, areawidth, areaheight)
        assert apply_values(viewer, dc.replace(viewer.values, figwidthscale=figwidthscale)) is None
        figwidth, figheight = viewer.fig.get_size_inches()
        # the scale has 2 decimals, thus the shape of the figure differs a little from the shape of the area
        assert np.isclose(figwidth / figheight, areawidth / areaheight, rtol=0.01)
        assert f" --figwidthscale {figwidthscale:g}" in viewer.get_command()


def test_maxseriescount_cuts_the_fixedionlist() -> None:
    """A smaller -maxseriescount removes the last entries of -fixedionlist, and a list alone keeps each entry.

    Before the correction, a -fixedionlist kept each entry, and -maxseriescount applied only to the names in the
    order of the flux.
    """
    viewer = make_headless_viewer([str(modelpath_classic_3d), "-t", "4", "--showemission", "--interactive"])
    series = viewer.get_drawn_series()
    assert len(series) > 2
    assert apply_values(viewer, dc.replace(viewer.values, fixedionlist=series[::-1], maxseriescount=2)) is None
    assert viewer.get_drawn_series() == series[::-1][:2]

    for extraargs, maxseriescount in (("", 20), ("-maxseriescount 5", 5)):
        argsraw = [str(modelpath_classic_3d), "--showemission", "-fixedionlist", *(["Fe II"] * 20), *extraargs.split()]
        args = at.misc.parse_cli_args(plotspectra.addargs, None, None, argsraw)
        plotspectra.resolve_plot_args(args)
        assert args.maxseriescount == maxseriescount


def test_interactive_render_changes_nothing_until_the_window_shows_the_plot() -> None:
    """A render draws a new figure and changes nothing of the viewer. Only the function that it returns shows the plot.

    The worker thread runs the render while the window reads the viewer. Thus a render that changed the figure or the
    frames gave the window a plot in progress. The spectrum viewer drew in the thread of the window for this reason.
    """
    viewer = make_headless_viewer([str(modelpath), "-t", "300", "--interactive"])
    oldfig, oldaxes = viewer.fig, viewer.axes
    newvalues = viewer.step_time(1)
    assert newvalues is not None

    show_plot = viewer.render(newvalues)
    assert viewer.fig is oldfig
    assert viewer.axes is oldaxes

    assert show_plot() is None
    assert viewer.fig is not oldfig
    assert viewer.fig.canvas is oldfig.canvas, "the canvas of the window must show the new figure"


def test_interactive_assertion_of_plotspectra_is_a_rejection() -> None:
    """An AssertionError of plotspectra is a rejection, and the viewer keeps the old values.

    The viewer caught fewer errors than the CLI, thus the error left the window with the rejected values.
    """
    viewer = make_headless_viewer([str(modelpath), "-t", "300", "--interactive"])
    oldvalues = viewer.values
    # resolve_plot_args asserts that the command does not give both kinds of viewing angle
    newvalues = dc.replace(oldvalues, otheroptions=(("-plotvspecpol", ("0",)), ("-plotviewingangle", ("0",))))
    assert viewer.get_rejection(newvalues) is not None
    assert apply_values(viewer, newvalues) is not None
    assert viewer.values == oldvalues


def test_interactive_rejected_frames_leave_a_plot_on_the_figure() -> None:
    """An error of the new frames empties the figure, and the next plot must make new frames on it.

    The viewer kept the key of the old frames, thus each later plot went to axes that the figure did not hold.
    """
    viewer = make_headless_viewer([str(modelpath), "-t", "290", "--interactive"])
    assert apply_values(viewer, dc.replace(viewer.values, otheroptions=(("-figscale", ("-10",)),))) is not None
    assert apply_values(viewer, viewer.step_time(1) or viewer.values) is None
    assert viewer.fig.axes, "the figure must hold the frames of the plot"


def test_interactive_redraw_matches_a_new_plot() -> None:
    """A plot that the viewer draws again gives the same pixels as a new plot of the same command.

    The viewer keeps the tick objects and the title position from one plot to the next.
    """
    viewer = make_headless_viewer([str(modelpath_classic_3d), "-t", "4", "--interactive"])
    later = viewer.step_time(2)
    assert later is not None
    for values in (
        dc.replace(viewer.values, yscale="log", logscalex=True),
        dc.replace(viewer.values, showemission=True),
        dc.replace(later, showemission=True, maxseriescount=4),
    ):
        assert apply_values(viewer, values) is None
    newviewer = make_headless_viewer([*shlex.split(viewer.get_command())[2:], "--interactive"])
    canvases = [viewer.fig.canvas, newviewer.fig.canvas]
    pixels = []
    for canvas in canvases:
        assert isinstance(canvas, FigureCanvasAgg)
        canvas.draw()
        pixels.append(np.asarray(canvas.buffer_rgba()).copy())
    assert np.array_equal(pixels[0], pixels[1])


def test_interactive_tick_labels_come_back_after_hidexticklabels() -> None:
    """The tick labels come back when the user removes --hidexticklabels from the table.

    cla() keeps the tick parameters, and the plot hides the labels only when the command gives the option.
    """
    viewer = make_headless_viewer([str(modelpath), "-t", "300", "--interactive"])
    for otheroptions, labelsvisible in (((("--hidexticklabels", ()),), False), ((), True)):
        assert apply_values(viewer, dc.replace(viewer.values, otheroptions=otheroptions)) is None
        viewer.fig.canvas.draw()
        assert viewer.axes[0].xaxis.get_major_ticks()[0].label1.get_visible() is labelsvisible


def test_interactive_status_line_gives_the_error() -> None:
    """The status line gives the error of argparse, and not the usage line that argparse prints before it."""
    viewer = make_headless_viewer([str(modelpath), "-t", "300", "--interactive"])
    rejection = viewer.get_rejection(dc.replace(viewer.values, deltax="20", otheroptions=(("-deltalambda", ("5",)),)))
    assert rejection is not None
    assert rejection.startswith("Argument --deltalambda")


def test_interactive_zero_xmin_keeps_its_side_in_a_new_unit() -> None:
    """A minimum wavelength of 0 becomes a range of frequencies above the other limit, and not below it.

    The zero limit took the default limit at the same position, thus the range went to the other side of 5000 Å.
    """
    viewer = make_headless_viewer([str(modelpath), "-t", "300", "--interactive"])
    values = dc.replace(viewer.values, xmin="0", xmax="5000")
    inhertz = interactive.convert_xunit(values, "hz", gamma=False)
    assert np.isclose(float(inhertz.xmin), at.constants.c_ang_per_s / 5000.0, rtol=1e-3, atol=0.0)
    # the range goes from 5000 Å to shorter wavelengths, which are higher frequencies
    assert float(inhertz.xmax) > float(inhertz.xmin)
    # a change between two units of wavelength keeps a minimum of 0
    inmicrons = interactive.convert_xunit(values, "micron", gamma=False)
    assert float(inmicrons.xmin) == 0.0


def test_interactive_single_time_ends_at_the_next_start() -> None:
    """A single time selects its timestep by the rule of get_timestep_of_timedays: a timestep ends at the next start.

    The end of a timestep can be a little after the start of the next one, and 2 then selected the next timestep.
    """
    tmids, tstarts, tends = [1.75, 2.25], [1.5, 2.0], [2.000008, 2.5]
    timedays = interactive.get_snapped_timedays_argument(tmids, tstarts, tends, 0, 0)
    assert tstarts[0] <= float(timedays) < tstarts[1]


def test_interactive_reference_name_finds_the_picked_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The command gives a bundled reference by its name only when plotspectra finds that same file by the name.

    plotspectra searches the working folder first, thus a file of the same name there took the place of the file
    that the user selected.
    """
    name = "sn2011fe_PTF11kly_20120822_norm.txt"
    bundledfile = plotspectra.find_reference_spectrum_file_or_none(name)
    assert bundledfile is not None
    monkeypatch.chdir(tmp_path)
    find = plotspectra.find_reference_spectrum_file_or_none
    assert viewerseries.get_reference_token(str(bundledfile), find) == name
    (tmp_path / name).write_text("1000 1\n2000 2\n")
    assert viewerseries.get_reference_token(str(bundledfile), find) == str(bundledfile)


def test_interactive_valid_times_of_each_model() -> None:
    """The time controls stay inside the times that are valid for every model of the command.

    plotspectra checks the times of each model, and the viewer took the valid times of the first model only.
    """
    # each call of the mock gives the valid range of the next model
    ranges = [(None, 260.0, 330.0), (None, 270.0, 320.0)]
    with mock.patch.object(interactive, "get_escaped_arrivalrange", side_effect=ranges):
        viewer = make_headless_viewer([str(modelpath), str(modelpath), "-t", "300", "--interactive"])
    assert viewer.grid.timebounds == (270.0, 320.0)


def test_interactive_command_of_a_dispatcher_call() -> None:
    """A call of the dispatcher from Python code gives its own words to the viewer, and not the words of sys.argv."""
    from artistools.__main__ import main as dispatcher_main

    with (
        mock.patch.object(viewerapplication, "run_viewer_application") as mockrunviewer,
        mock.patch.object(sys, "argv", ["myscript.py", "--flag", "x"]),
    ):
        dispatcher_main(argsraw=["plotspectra", str(modelpath), "-t", "300", "--interactive"])
    mockrunviewer.assert_called_once()
    assert mockrunviewer.call_args.args[2:] == (
        interactive.open_window,
        [str(modelpath), "-t", "300", "--interactive"],
        interactive.DOCUMENT_TYPES,
    )


def test_interactive_unlock_gives_the_default_count() -> None:
    """A count that came from the length of -fixedionlist becomes the default count when the lock ends."""
    viewer = make_headless_viewer([str(modelpath), "-t", "300", "--interactive"])
    locked = dc.replace(viewer.values, fixedionlist=("Fe II", "Co II"), maxseriescount=2)
    assert interactive.remove_series_lock(locked).maxseriescount == plotspectra.DEFAULT_MAXSERIESCOUNT
    assert interactive.remove_series_lock(dc.replace(locked, maxseriescount=5)).maxseriescount == 5


def test_interactive_data_source() -> None:
    """The data source gives --frompackets, Auto names the files that it selects, and Text files refuses the packets.

    The --frompackets box showed only the flag of the user. An option that needs the packets, e.g. -groupby element,
    made plotspectra read them with the box empty.
    """
    viewer = make_headless_viewer([str(modelpath_classic_3d), "-t", "4", "--frompackets", "--interactive"])
    assert viewer.values.datasource == "packets"
    assert not viewer.values.otheroptions
    assert "--frompackets" in shlex.split(viewer.get_command())

    assert apply_values(viewer, dc.replace(viewer.values, datasource="auto")) is None
    assert "--frompackets" not in shlex.split(viewer.get_command())
    assert interactive.get_packets_reason(viewer.get_plot_tokens()) is None
    elementvalues = dc.replace(viewer.values, showemission=True, groupby="element")
    assert interactive.get_packets_reason(viewer.get_plot_tokens(elementvalues)) == "--groupby element"

    assert apply_values(viewer, dc.replace(viewer.values, datasource="text")) is None
    textelement = dc.replace(elementvalues, datasource="text")
    oldvalues = viewer.values
    for message in (viewer.get_rejection(textelement), apply_values(viewer, textelement)):
        assert message is not None
        assert message.startswith("--groupby element needs the packets files")
    assert viewer.values == oldvalues


def test_interactive_direction_and_bin_controls() -> None:
    """The controls of the viewing direction and the bins take the options of the user, and give each option one time.

    The deprecated --average-every-tenth-viewing-angle becomes --average-over-phi-angle, and the table of the other
    options holds none of these options.
    """
    viewer = make_headless_viewer([
        str(modelpath_classic_3d),
        "-t",
        "4",
        "--average-every-tenth-viewing-angle",
        "--plotviewingangle",
        "10",
        "--dlogx",
        "0.002",
        "--normalised",
        "--interactive",
    ])
    values = viewer.values
    assert (values.directionkind, values.directionbins, values.deltalogx) == ("phi", (10,), "0.002")
    assert values.normalised
    assert not values.otheroptions
    command = shlex.split(viewer.get_command())
    for option in ("--average-over-phi-angle", "--plotviewingangle", "--deltalogx", "--normalised"):
        assert command.count(option) == 1
    assert "--average-every-tenth-viewing-angle" not in command
    assert command[command.index("--plotviewingangle") + 1] == "10"

    choices = viewercore.get_direction_choices(modelpath_classic_3d, "theta", usedegrees=False)
    assert [dirbin for dirbin, _ in choices] == list(range(10))
    assert apply_values(viewer, dc.replace(values, directionkind="theta", directionbins=(3,), deltalogx="")) is None
    command = shlex.split(viewer.get_command())
    assert "--average-over-theta-angle" in command
    assert not {"--average-over-phi-angle", "--deltalogx"} & set(command)

    # the legend gives the angles of the direction, and the command keeps --usedegrees only with a direction
    assert apply_values(viewer, dc.replace(viewer.values, usedegrees=True)) is None
    assert "--usedegrees" in shlex.split(viewer.get_command())
    legend = viewer.axes[0].get_legend()
    assert legend is not None
    assert any("°" in text.get_text() for text in legend.get_texts())

    assert apply_values(viewer, dc.replace(viewer.values, directionkind="", directionbins=())) is None
    command = set(shlex.split(viewer.get_command()))
    assert not {"--plotviewingangle", "--average-over-theta-angle", "--usedegrees"} & command


def test_interactive_option_rows() -> None:
    """The table of the window reads each form of an option that argparse accepts, and each row keeps its values."""
    parser = viewercore.make_parser(plotspectra.addargs)
    tokens = [
        "--dx",
        "5",
        "--label",
        "a b",
        "c",
        "--filtersavgol",
        "5",
        "2",
        "--normalised",
        "--title=My plot",
        "-dpi300",
    ]
    rows, othertokens = viewercore.split_option_rows(parser, [*tokens, "--", "rest"])
    assert rows == (
        ("--deltax", ("5",)),
        ("--label", ("a b", "c")),
        ("--filtersavgol", ("5", "2")),
        ("--normalised", ()),
        ("--title", ("My plot",)),
        ("--dpi", ("300",)),
    )
    assert othertokens == ["--", "rest"]

    actions = {
        action.option_strings[0]: action
        for action in viewercore.get_table_actions(
            parser, interactive.CONTROLLED_DESTS | interactive.TABLE_EXCLUDED_DESTS
        )
    }
    # the other controls of the window set these options, and a list of times draws more than one plot
    assert (
        not {"--timedays", "--xmin", "--groupby", "--yvariable", "--plotviewingangle", "--timedayslist", "-h"}
        & actions.keys()
    )
    # no option of the table has choices now, but a new option with choices gets a list in the table
    allactions = viewercore.get_actions_by_flag(parser)
    kinds = {flag: viewercore.get_option_kind(allactions[flag]) for flag in ("--notitle", "--yvariable", "--dpi")}
    assert kinds == {"--notitle": "flag", "--yvariable": "choice", "--dpi": "int"}
    assert [viewercore.get_option_kind(actions[flag]) for flag in ("--filtersavgol", "--label", "--title")] == [
        "values",
        "list",
        "text",
    ]
    # an option with no default needs a value from the user before the command can give it. Export Figure asks for
    # -dpi, thus the table does not offer it
    assert "--dpi" not in actions
    assert viewercore.get_default_tokens(allactions["--dpi"]) == ("250",)
    assert viewercore.get_default_tokens(actions["--title"]) is None


def test_interactive_other_options_reach_the_command() -> None:
    """The command contains each row of the table, and a row that plotspectra rejects keeps the old options."""
    viewer = make_headless_viewer([str(modelpath), "-t", "300", "--filtermovingavg", "3", "--interactive"])
    assert viewer.values.otheroptions == (("--filtermovingavg", ("3",)),)

    otheroptions = (("--filtermovingavg", ("5",)), ("--title", ("A title",)))
    assert apply_values(viewer, dc.replace(viewer.values, otheroptions=otheroptions)) is None
    command = viewer.get_command()
    assert " --filtermovingavg 5 --title 'A title'" in command
    assert viewer.axes[0].get_title() == "A title"

    # a ratio of Stokes parameters is a different plot from the one that the window shows
    message = apply_values(viewer, dc.replace(viewer.values, otheroptions=(("--stokesparam", ("Q/I",)),)))
    assert message is not None
    assert "--stokesparam Q/I" in message
    assert viewer.values.otheroptions == otheroptions


def test_interactive_lock_series() -> None:
    """A locked list of series keeps the colour of each series when the time changes."""
    viewer = make_headless_viewer([str(modelpath_classic_3d), "-t", "4", "--showemission", "--interactive"])

    def get_series_colours() -> dict[str, tuple[float, float, float, float]]:
        legend = viewer.axes[0].get_legend()
        assert legend is not None
        return {
            text.get_text(): mplcolors.to_rgba(handle.get_facecolor())
            for text, handle in zip(legend.get_texts(), legend.legend_handles, strict=True)
            if isinstance(handle, mpatches.Patch)
        }

    locked = viewer.get_drawn_series()
    assert len(locked) > 1
    assert apply_values(viewer, dc.replace(viewer.values, fixedionlist=locked)) is None
    tokens = shlex.split(viewer.get_command())
    assert tokens[tokens.index("--fixedionlist") + 1 :] == list(locked)
    colours = get_series_colours()

    later = viewer.step_time(3)
    assert later is not None
    assert apply_values(viewer, later) is None
    latercolours = get_series_colours()
    shared = set(colours) & set(latercolours)
    assert shared
    assert all(colours[name] == latercolours[name] for name in shared)


def test_interactive_time_stays_inside_the_models_of_the_list(tmp_path: Path) -> None:
    """A model that the user adds can have a shorter valid time, and the time controls then stay inside it.

    The viewer read the valid times of the models of the command only. After Add Model, the first time of the slider
    was before the valid range of the new model, and plotspectra rejected it.
    """
    # a copy of the model with a smaller maximum velocity gives a longer range of arrival times
    widemodel = tmp_path / "widemodel"
    widemodel.mkdir()
    for path in modelpath.iterdir():
        if path.name != "model.txt":
            (widemodel / path.name).symlink_to(path.absolute())
    modeltext = (modelpath / "model.txt").read_text(encoding="utf-8")
    (widemodel / "model.txt").write_text(modeltext.replace("8000.", "1000.", 1), encoding="utf-8")

    fig = mplfig.Figure()
    FigureCanvasAgg(fig)
    viewer = interactive.SpectrumViewer([str(widemodel), "-t", "300", "--interactive"], fig)
    widestart = viewer.grid.timebounds[0]
    viewer.load_models((str(widemodel), str(modelpath)))
    assert viewer.grid.timebounds[0] > widestart
    firsttime = viewer.move_to_end(last=False)
    assert apply_values(viewer, dc.replace(firsttime, spectra=(str(widemodel), str(modelpath)))) is None


def test_interactive_spectrum_path_is_the_same_for_each_spelling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two spellings of one spectrum give the same path, thus the list of spectra does not show it two times.

    A drop of the working folder added it again next to ".", and the plot then drew the same model two times.
    """
    monkeypatch.chdir(modelpath)

    def get_full_path(path: str) -> Path:
        return viewerseries.get_series_full_path(interactive.REFERENCE_DATA, path)

    assert get_full_path(".") == get_full_path(str(modelpath.absolute()))
    assert get_full_path(".") != get_full_path(str(modelpath.parent))


def test_interactive_switch_between_r_packets_and_gamma_packets() -> None:
    """The packet control switches the plot between the r-packets and the gamma packets, with the units of each.

    --gamma was an option of the table, thus the x unit, the x range, and the grouping kept the defaults of the
    r-packets after a switch.
    """
    viewer = make_headless_viewer([str(modelpath_classic_3d), "-t", "4", "--interactive"])
    assert not viewer.values.gamma
    assert apply_values(viewer, interactive.set_packet_type(viewer.values, gamma=True)) is None
    tokens = viewer.get_plot_tokens()
    assert "--gamma" in tokens
    assert "-xunit" not in tokens, "keV is the default x unit of a gamma-ray spectrum"
    assert viewer.values.xunit == "kev"
    assert apply_values(viewer, interactive.set_packet_type(viewer.values, gamma=False)) is None
    assert "--gamma" not in viewer.get_plot_tokens()
    assert viewer.values.xunit == "angstrom"
    # a command with --gamma starts in the mode of the gamma packets, and the option table does not show --gamma
    gammaviewer = make_headless_viewer([str(modelpath_classic_3d), "-t", "4", "--gamma", "--interactive"])
    assert gammaviewer.values.gamma
    assert all(flag != "--gamma" for flag, _ in gammaviewer.values.otheroptions)
    # the test model has packets and no gamma_spec.out, thus plotspectra reads the packets of its gamma-ray spectrum.
    # The viewer set --frompackets for the gamma packets, and that flag stayed after the switch back to the r-packets
    packetsviewer = make_headless_viewer([str(modelpath), "-t", "300", "--interactive"])
    assert packetsviewer.grid.hasgammaspectrum
    gammavalues = interactive.set_packet_type(packetsviewer.values, gamma=True)
    assert gammavalues.datasource == "auto"
    assert (
        interactive.get_packets_reason(packetsviewer.get_plot_tokens(gammavalues)) == "--gamma with no gamma_spec.out"
    )
    assert apply_values(packetsviewer, gammavalues) is None
    assert apply_values(packetsviewer, interactive.set_packet_type(packetsviewer.values, gamma=False)) is None
    assert packetsviewer.values.datasource == "auto"
    assert "--frompackets" not in packetsviewer.get_plot_tokens()
    # a model folder with neither file has no gamma-ray spectrum, thus the window disables the gamma packets
    assert not interactive.has_gamma_spectrum([at.get_path("testdata") / "test-classicmode_1d"])


def test_interactive_gamma_spectrum_needs_one_source_for_all_models(tmp_path: Path) -> None:
    """The gamma packets need gamma_spec.out in each model folder, or the packets of each model folder.

    plotspectra reads the packets of each model folder when one model folder has no gamma_spec.out. A list with
    gamma_spec.out alone in one model folder and packets alone in the other model folder enabled the gamma packets,
    and the plot then failed.
    """
    specmodel, packetsmodel = tmp_path / "specmodel", tmp_path / "packetsmodel"
    for modelfolder, filename in ((specmodel, "gamma_spec.out"), (packetsmodel, "packets00_0000.out")):
        modelfolder.mkdir()
        (modelfolder / filename).write_text("", encoding="utf-8")
    assert interactive.has_gamma_spectrum([specmodel])
    assert interactive.has_gamma_spectrum([packetsmodel])
    assert not interactive.has_gamma_spectrum([specmodel, packetsmodel])


def test_interactive_direction_kinds_follow_the_first_model() -> None:
    """The kinds of viewing direction come from the first model of the list, also after a change of the list.

    The kinds came from the first model of the command, thus a new first model with no vpkt.txt kept -plotvspecpol.
    """
    viewer = make_headless_viewer([str(at.get_path("testdata") / "vpktcontrib"), "--interactive"])
    assert "vpkt" in viewer.grid.directionkinds
    viewer.load_models([str(modelpath), str(at.get_path("testdata") / "vpktcontrib")])
    assert "vpkt" not in viewer.grid.directionkinds


def test_viewer_keys_the_models_by_the_tokens_of_the_list() -> None:
    """A path with a trailing slash must give the model of its row, or the window stops at the start."""
    fig = mplfig.Figure()
    FigureCanvasAgg(fig)
    viewer = interactive.SpectrumViewer([f"{modelpath}/", "-t", "300"], fig)
    assert set(viewer.grid.modeltimes) == set(viewer.values.spectra)
    assert viewer.grid.loadkey == (viewer.values.spectra, viewer.values.timegrid)


def test_viewer_command_keeps_the_time_grid_of_a_later_model() -> None:
    """A command made on the timesteps of a later model must open with the same time.

    The command holds no choice of "Timesteps of". The viewer snapped the time to the timesteps of the first model,
    thus a copied command or a restored session drew a different time range.
    """
    models = [
        str(at.get_path("testdata") / "test-classicmode_3d"),
        str(at.get_path("testdata") / "test-classicmode_1d"),
    ]

    def make(tokens: list[str]) -> interactive.SpectrumViewer:
        fig = mplfig.Figure()
        FigureCanvasAgg(fig)
        return interactive.SpectrumViewer(tokens, fig)

    viewer = make([*models, "-t", "4.5-5"])
    viewer.values = interactive.set_models(viewer, viewer.values.spectra, models[1])
    command = viewer.get_command()
    reopened = make([*shlex.split(command)[2:], "--interactive"])
    assert reopened.get_command() == command
    assert reopened.values.timegrid == models[1]


def test_interactive_command_reads_one_grid_of_the_models() -> None:
    """The worker thread makes the command from the timestep grid while the window can load the models again.

    A new model replaced the grid during a plot, and the worker then read a timestep of the old grid in the new grid.
    The command must read the grid one time, thus a load during the command has no effect on it.
    """
    viewer = make_headless_viewer([str(modelpath), "-t", "290", "--interactive"])
    expectedtokens = viewer.get_plot_tokens()
    oldgrid = viewer.grid
    get_grid_selection = interactive.get_grid_selection

    def select_then_load(grid: interactive.TimestepGrid, values: interactive.ControlValues) -> tuple[int, int]:
        viewer.grid = oldgrid._replace(tmids=oldgrid.tmids[:1], tstarts=oldgrid.tstarts[:1], tends=oldgrid.tends[:1])
        return get_grid_selection(grid, values)

    with mock.patch.object(interactive, "get_grid_selection", side_effect=select_then_load):
        assert viewer.get_plot_tokens() == expectedtokens
    assert len(viewer.grid.tmids) == 1


def test_interactive_packet_type_drops_a_bin_width_of_a_different_unit() -> None:
    """A new packet type with a new x unit drops -deltax, as a new x unit does, and keeps a -deltax of the same unit.

    The window gave the gamma-ray spectrum bins of 20 keV from a -deltax of 20 Å.
    """
    viewer = interactive.SpectrumViewer(
        [str(modelpath), "-t", "300", "-deltax", "20", "--interactive"], mplfig.Figure()
    )
    gammavalues = interactive.set_packet_type(viewer.values, gamma=True)
    assert (gammavalues.xunit, gammavalues.deltax) == ("kev", "")
    assert not interactive.set_packet_type(gammavalues, gamma=False).deltax
    kevvalues = dc.replace(viewer.values, xunit="kev")
    assert interactive.set_packet_type(kevvalues, gamma=True).deltax == "20"


@pytest.mark.parametrize(("firsttimestep", "lasttimestep"), [(54, 55), (40, 41), (40, 42), (40, 44)])
def test_interactive_continuous_range_keeps_the_days_of_the_snapped_range(
    firsttimestep: int, lasttimestep: int
) -> None:
    """A change from Snap to Timesteps to Continuous keeps the days of the whole timesteps, and a snap gives them back.

    The continuous range went from the middle of the first timestep to the middle of the last, thus it was one
    timestep shorter, and the snap back lost a timestep.
    """
    viewer = interactive.SpectrumViewer(
        [str(modelpath), "-timestep", f"{firsttimestep}-{lasttimestep}", "--interactive"], mplfig.Figure()
    )
    assert interactive.get_grid_selection(viewer.grid, viewer.values) == (firsttimestep, lasttimestep)
    continuous = viewer.clamp_time(interactive.get_continuous_values(viewer, viewer.values))
    assert continuous.notimeclamp
    grid = viewer.grid
    assert np.isclose(continuous.width, grid.tends[lasttimestep] - grid.tstarts[firsttimestep], rtol=0.01)
    snapped = viewer.snap(continuous, *interactive.get_grid_selection(viewer.grid, continuous))
    assert interactive.get_grid_selection(viewer.grid, snapped) == (firsttimestep, lasttimestep)


def test_interactive_reload_reads_the_data_source_again(tmp_path: Path) -> None:
    """Reload Data must read again which files Auto selects, e.g. after exspec writes gamma_spec.out.

    The window kept the result of the old files, thus the data source box stayed at "Auto (packets)".
    """
    from artistools.viewertools.window import clear_output_caches_of_model

    modelfolder = tmp_path / "model"
    modelfolder.mkdir()
    for path in modelpath.iterdir():
        (modelfolder / path.name).symlink_to(path)
    assert not (modelfolder / "gamma_spec.out").exists()
    viewer = interactive.SpectrumViewer([str(modelfolder), "-t", "300", "--gamma", "--interactive"], mplfig.Figure())
    assert interactive.get_auto_reason(viewer, viewer.values) == "--gamma with no gamma_spec.out"
    (modelfolder / "gamma_spec.out").write_bytes((modelpath / "spec.out").read_bytes())
    # Reload Data clears the caches of the files, then it reads the models again
    clear_output_caches_of_model(modelfolder, estimatorsonly=False)
    viewer.load_models(viewer.values.spectra, viewer.values.timegrid)
    assert interactive.get_auto_reason(viewer, viewer.values) is None


def test_interactive_animation_keeps_the_time_grid_of_its_start() -> None:
    """The commands of Export Animation come from the time grid at the start, also when the window reads new models.

    Add Model during an export gave the later frames the positions of the old grid on the new grid.
    """
    viewer = interactive.SpectrumViewer([str(modelpath), "-t", "300", "--interactive"], mplfig.Figure())
    count, get_frame_tokens = interactive.get_animation_frames(viewer)
    firsttokens, lasttokens = get_frame_tokens(0), get_frame_tokens(count - 1)
    viewer.load_models([str(modelpath_classic_3d)])
    assert len(viewer.grid.validtimesteps) < count
    assert get_frame_tokens(0) == firsttokens
    assert get_frame_tokens(count - 1) == lasttokens


def test_interactive_time_field_takes_finite_numbers() -> None:
    """A time field must refuse "nan" and "inf", which float() reads and which a count of timesteps cannot take."""
    assert interactive.read_finite_number("12.5") == 12.5
    for text in ("nan", "inf", "-inf", "abc"):
        with pytest.raises(ValueError, match=r"finite|could not convert"):
            interactive.read_finite_number(text)
