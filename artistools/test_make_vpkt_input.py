import argparse
import dataclasses as dc
import itertools
import re
import typing as t
from pathlib import Path

import pytest

import artistools as at


def test_make_vpkt_input_default_contents() -> None:
    """The default vpkt.txt must keep the exact layout ARTIS parses, field by field."""
    expected = (
        "3\n"  # number of viewing directions
        "1 0 -1\n"  # costheta of each direction
        "0 0 0\n"  # phi of each direction
        "0 \n"  # no opacity exclusions
        "0 0.2 1.5\n"  # override_tminmax off, then the time window
        "0\n"  # no custom wavelength ranges
        "1 100\n"  # override thick cell tau, and the threshold
        "10\n"  # tau_max_vpkt
        "0\n"  # velocity grid map off
        "0.2 1.5\n"  # velocity grid map time range
        "1 3500 6000"  # one wavelength range for the velocity grid map
    )
    assert at.make_vpkt_input.format_vpkt_input(at.misc.modelinfo.VpktConfig()) == expected


def test_make_vpkt_input_optional_blocks() -> None:
    """The opacity exclusion and custom wavelength blocks must be prefixed by their own counts."""
    config = at.misc.modelinfo.VpktConfig(
        directions_costheta_phi=[(-1, 0), (0.5, 90)],
        opacityexclusions=[0, -1, 26],
        custom_lambda_ranges=[(3500, 6000), (10000, 12000)],
        override_tminmax=True,
        vgrid_on=True,
        override_thickcell_tau=False,
        tau_max_vpkt=7.5,
    )
    contents = at.make_vpkt_input.format_vpkt_input(config).splitlines()

    assert contents[0] == "2"
    assert contents[1] == "-1 0.5"
    assert contents[2] == "0 90"
    assert contents[3] == "1 3 0 -1 26"
    assert contents[4] == "1 0.2 1.5"
    assert contents[5] == "1 2 3500 6000 10000 12000"
    assert contents[6] == "0 100"
    assert contents[7] == "7.5"
    assert contents[8] == "1"


def test_make_vpkt_input_roundtrip() -> None:
    """Parsing a written file must recover exactly the settings it was written from."""
    for config in (
        at.misc.modelinfo.VpktConfig(),
        at.misc.modelinfo.VpktConfig(
            directions_costheta_phi=[(-1, 0), (0.5, 90)],
            opacityexclusions=[0, -1, 26],
            custom_lambda_ranges=[(3500, 6000), (10000, 12000)],
            override_tminmax=True,
            vgrid_on=True,
            override_thickcell_tau=False,
            tau_max_vpkt=7.5,
            vspec_tmin_in_days=1.25,
        ),
    ):
        assert at.misc.parse_vpkt_input(at.make_vpkt_input.format_vpkt_input(config)) == config


def test_make_vpkt_input_rejects_inconsistent_file() -> None:
    """A truncated or out-of-range file must be reported, not silently accepted."""
    contents = at.make_vpkt_input.format_vpkt_input(at.misc.modelinfo.VpktConfig())
    truncated = "\n".join(contents.splitlines()[:5])
    with pytest.raises(ValueError, match="ended while reading"):
        at.misc.parse_vpkt_input(truncated)

    # a file ARTIS would reject still loads, so it can be repaired, but reports the problem
    badcostheta = at.misc.parse_vpkt_input(contents.replace("1 0 -1", "1 0 -2", 1))
    assert "outside [-1, 1]" in str(at.make_vpkt_input.fatal_config_error(badcostheta))
    with pytest.raises(ValueError, match="outside"):
        at.make_vpkt_input.format_vpkt_input(badcostheta)


def test_make_vpkt_input_matches_artis_token_reader() -> None:
    """ARTIS reads vpkt.txt with fscanf, which ignores line breaks, so the parser must too."""
    config = at.misc.modelinfo.VpktConfig(
        directions_costheta_phi=[(0.5, 90)], opacityexclusions=[0, -1], custom_lambda_ranges=[(4000, 7000)]
    )
    contents = at.make_vpkt_input.format_vpkt_input(config)

    allonelongline = " ".join(contents.split())
    assert at.misc.parse_vpkt_input(allonelongline) == config

    onetokenperline = "\n".join(contents.split())
    assert at.misc.parse_vpkt_input(onetokenperline) == config


def test_make_vpkt_input_velocity_grid_ranges_roundtrip() -> None:
    """ARTIS reads as many velocity grid ranges as the count declares, so every one must be written."""
    config = at.misc.modelinfo.VpktConfig(
        vgrid_on=True, vgrid_lambda_ranges=[(3500, 6000), (6000, 9000), (9000, 10000)]
    )
    contents = at.make_vpkt_input.format_vpkt_input(config)

    assert contents.splitlines()[-1] == "3 3500 6000 6000 9000 9000 10000"
    assert at.misc.parse_vpkt_input(contents) == config


def test_make_vpkt_input_accepts_file_without_velocity_grid_block() -> None:
    """ARTIS only reads the velocity grid block when the map is on, so a file may legitimately omit it."""
    contents = at.make_vpkt_input.format_vpkt_input(at.misc.modelinfo.VpktConfig())
    withoutvgridblock = "\n".join(contents.splitlines()[:9])

    config = at.misc.parse_vpkt_input(withoutvgridblock)
    assert not config.vgrid_on
    assert config.vgrid_lambda_ranges == [(3500.0, 6000.0)]


def test_make_vpkt_input_rejects_nonzero_first_opacity_choice() -> None:
    """ARTIS asserts opacityexclusions[0] == 0, so artistools must not be able to write such a file."""
    config = at.misc.modelinfo.VpktConfig(opacityexclusions=[26, 0])
    with pytest.raises(ValueError, match="first opacity choice must be 0"):
        at.make_vpkt_input.format_vpkt_input(config)


def test_make_vpkt_input_warns_outside_compiled_limits() -> None:
    """Bounds that ARTIS asserts against compile-time constants must warn rather than fail."""
    outsidetime = at.misc.modelinfo.VpktConfig(override_tminmax=True, vspec_tmin_in_days=0.2, vspec_tmax_in_days=1.5)
    assert any("time window" in warning for warning in at.make_vpkt_input.check_config(outsidetime))

    outsidelambda = at.misc.modelinfo.VpktConfig(custom_lambda_ranges=[(1000, 2000)])
    assert any("wavelength range" in warning for warning in at.make_vpkt_input.check_config(outsidelambda))

    assert not at.make_vpkt_input.check_config(at.misc.modelinfo.VpktConfig()), "the defaults must not warn"


def test_make_vpkt_input_cli_writes_file(tmp_path: Path) -> None:
    """The subcommand must honour -directions, including a negative leading costheta, and -outputfile."""
    outfile = tmp_path / "vpkt.txt"
    at.make_vpkt_input.main(argsraw=["-directions=-1,0 1,0", "-o", str(outfile), "--non-interactive"])

    lines = outfile.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "2"
    assert lines[1] == "-1 1"
    assert lines[2] == "0 0"


def test_make_vpkt_input_cli_keeps_existing_settings(tmp_path: Path) -> None:
    """Rerunning on an existing file must preserve settings that were not given on the command line."""
    outfile = tmp_path / "vpkt.txt"
    at.make_vpkt_input.main(argsraw=["-tau-max", "7.5", "-o", str(outfile), "--non-interactive"])
    assert at.misc.parse_vpkt_input(outfile.read_text(encoding="utf-8")).tau_max_vpkt == 7.5

    at.make_vpkt_input.main(argsraw=["-vspec-tmax", "9.0", "-o", str(outfile), "--non-interactive"])
    config = at.misc.parse_vpkt_input(outfile.read_text(encoding="utf-8"))
    assert config.vspec_tmax_in_days == 9.0
    assert config.tau_max_vpkt == 7.5, "the tau-max from the first run must survive the second"


def test_make_vpkt_input_interactive_edit() -> None:
    """An empty reply keeps the current value, and an invalid reply is asked again."""
    replies = iter([
        "",  # keep the default viewing directions
        "26 -1",  # rejected: the first opacity choice must be 0, so this question repeats
        "0 26 -1",  # set opacity choices
        "maybe",  # rejected, so this question repeats
        "yes",  # override_tminmax
        "",  # keep vspec_tmin
        "3.5",  # vspec_tmax
    ])
    # pad so the reply script does not have to track how many settings there are
    replies = itertools.chain(replies, itertools.repeat(""))
    asked: list[str] = []

    def fakeprompt(text: str) -> str:
        asked.append(text)
        return next(replies)

    config = at.make_vpkt_input.edit_config_interactively(at.misc.modelinfo.VpktConfig(), promptfunc=fakeprompt)

    assert config.directions_costheta_phi == [(1.0, 0.0), (0.0, 0.0), (-1.0, 0.0)]
    assert config.opacityexclusions == [0, 26, -1]
    assert config.override_tminmax
    assert config.vspec_tmin_in_days == 0.2
    assert config.vspec_tmax_in_days == 3.5
    # the rejected reply must have caused the same question to be asked twice
    assert sum(question.startswith("Restrict virtual packets") for question in asked) == 2
    assert "[1,0 0,0 -1,0]" in asked[0], "the prompt must show the current value"


def test_make_vpkt_input_interactive_clears_list() -> None:
    """A single '-' must clear a list-valued setting."""
    config = at.misc.modelinfo.VpktConfig(opacityexclusions=[26])
    replies = itertools.chain(iter(["", "-"]), itertools.repeat(""))
    config = at.make_vpkt_input.edit_config_interactively(config, promptfunc=lambda _: next(replies))

    assert config.opacityexclusions == []


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"opacityexclusions": [0, -5]}, "below -4"),
        ({"custom_lambda_ranges": [(0.0, 6000.0)]}, "lambdamin > 0"),
        ({"tau_max_vpkt": 0.0}, "tau_max_vpkt 0 must be more than zero"),
        ({"cell_is_optically_thick_vpkt": -1.0}, "thick cell optical depth -1"),
        ({"override_tminmax": True, "vspec_tmin_in_days": 4.0, "vspec_tmax_in_days": 4.0}, "time window"),
        ({"vgrid_on": True, "tmin_vgrid_in_days": 2.0, "tmax_vgrid_in_days": 1.0}, "velocity grid map time range"),
        ({"vgrid_on": True, "vgrid_lambda_ranges": []}, "at least one wavelength range"),
        ({"vgrid_on": True, "vgrid_lambda_ranges": [(0.0, 6000.0)]}, "lambdamin > 0"),
    ],
)
def test_make_vpkt_input_refuses_each_fatal_setting_of_artis(changes: dict[str, t.Any], reason: str) -> None:
    """Each setting that read_vpktparameterfile() in vpkt.cc stops on must give an error, not a file.

    The command wrote a vpkt.txt with such a setting, and ARTIS then stopped at the start of the run.
    """
    config = dc.replace(at.misc.modelinfo.VpktConfig(), **changes)
    assert reason in str(at.make_vpkt_input.fatal_config_error(config))
    with pytest.raises(ValueError, match=re.escape(reason)):
        at.make_vpkt_input.format_vpkt_input(config)

    # ARTIS reads the ranges of the velocity grid map only when the map is on
    assert (
        at.make_vpkt_input.fatal_config_error(dc.replace(at.misc.modelinfo.VpktConfig(), vgrid_lambda_ranges=[]))
        is None
    )
    assert (
        at.make_vpkt_input.fatal_config_error(
            dc.replace(at.misc.modelinfo.VpktConfig(), vgrid_lambda_ranges=[(0.0, 6000.0)])
        )
        is None
    )


def test_make_vpkt_input_rejects_bad_arguments() -> None:
    for baddirection in ("1", "1,0,0", "north,0", "1.5,0"):
        with pytest.raises(argparse.ArgumentTypeError):
            at.make_vpkt_input.parse_directions(baddirection)

    for badrange in ("3500", "3500,6000,7000", "blue,red", "6000,3500"):
        with pytest.raises(argparse.ArgumentTypeError):
            at.make_vpkt_input.parse_lambda_range(badrange)

    for badbool in ("maybe", "", "2"):
        with pytest.raises(argparse.ArgumentTypeError):
            at.make_vpkt_input.parse_bool(badbool)
