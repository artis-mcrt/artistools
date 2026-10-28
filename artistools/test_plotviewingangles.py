import re
from pathlib import Path
from unittest import mock

import numpy as np
import polars as pl
import pytest

import artistools as at

modelpath_3d = at.get_path("testdata") / "testmodel_3d_10^3"


def read_data_file(path: Path) -> pl.DataFrame:
    """Read a table that --write_data wrote."""
    return pl.read_csv(path, separator=" ", null_values="nan", comment_prefix="#")


def test_plotviewingangles_writes_an_animation_of_each_direction_bin(tmp_path: Path) -> None:
    """The html file holds one frame of the animation for each of the 100 direction bins."""
    pytest.importorskip("plotly")
    outputfile = tmp_path / "viewingangles.html"
    at.plotviewingangles.main(argsraw=[str(modelpath_3d), "-o", str(outputfile)])

    framenames = set(re.findall(r'"name":"(\d\d)"', outputfile.read_text(encoding="utf-8")))
    assert framenames == {f"{dirbin:02d}" for dirbin in range(100)}


def test_plotviewingangles_write_data_holds_the_volume_and_the_lines(tmp_path: Path) -> None:
    """The tables of --write_data must give the density of each cell of the volume and the points of each line."""
    pytest.importorskip("plotly")
    import plotly.express as px
    import plotly.graph_objects as go

    outputfile = tmp_path / "viewingangles.html"
    with (
        mock.patch.object(go, "Volume", side_effect=go.Volume) as mockvolume,
        mock.patch.object(px, "line_3d", side_effect=px.line_3d) as mockline3d,
    ):
        at.plotviewingangles.main(argsraw=[str(modelpath_3d), "-o", str(outputfile), "--write_data"])

    dfvolume = read_data_file(tmp_path / "viewingangles.txt")
    volumekwargs = mockvolume.call_args.kwargs
    for colname, kwarg in (
        ("pos_x_mid_cm", "x"),
        ("pos_y_mid_cm", "y"),
        ("pos_z_mid_cm", "z"),
        ("rho_g_per_cm3", "value"),
    ):
        assert np.allclose(dfvolume[colname].to_numpy(), volumekwargs[kwarg], rtol=1e-12, atol=0)

    dflines = read_data_file(tmp_path / "viewingangles_directions.txt")
    dfdrawnlines = mockline3d.call_args.args[0]
    assert dflines["dirbin"].to_list() == [int(dirbin) for dirbin in dfdrawnlines["Angle-bin"]]
    for axis in ("x", "y", "z"):
        assert np.allclose(
            dflines[f"{axis}_cm"].to_numpy(), dfdrawnlines[f"{axis}_coord"].to_numpy(), rtol=1e-12, atol=0
        )
