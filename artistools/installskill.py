"""Install the artis-data skill. It tells a large language model (LLM) agent how to read the data of a model.

The command makes a link to the skill in the installed package. An update of artistools then also updates the
skill. With --copy, the command writes a copy. Install the skill again after each update of artistools.
"""

import argparse
import os
import shutil
import typing as t
from collections.abc import Sequence
from pathlib import Path

from artistools.misc import addarg_force
from artistools.misc import confirm_overwrite
from artistools.misc import exit_with_error
from artistools.misc import parse_cli_args
from artistools.misc import print_detail

SKILLNAME = "artis-data"


def get_skill_folder() -> Path:
    """Return the folder of the skill in the installed package."""
    return Path(__file__).parent / "data" / "skills" / SKILLNAME


def get_layout_skillsdir(layout: str) -> Path:
    """Return the skills folder of the user for a layout, e.g. ~/.agents/skills for "agents".

    If CLAUDE_CONFIG_DIR has a value, Claude Code uses that folder in place of ~/.claude.
    """
    if layout == "claude" and (claudeconfigdir := os.environ.get("CLAUDE_CONFIG_DIR")):
        return Path(claudeconfigdir).expanduser() / "skills"
    return Path.home() / f".{layout}" / "skills"


def is_in_folder(path: Path, folder: Path) -> bool:
    """Return True when the path is the folder or is below it, also for a different spelling of the folder."""
    return any(candidate.exists() and candidate.samefile(folder) for candidate in (path, *path.parents))


def is_in_uv_cache(path: Path) -> bool:
    """Return True when the path is in an archive of the uv cache, e.g. when uvx runs artistools."""
    return any(part.startswith("archive-v") and part.removeprefix("archive-v").isdigit() for part in path.parts)


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add the arguments that select the skills folder and the type of the installation."""
    foldergroup = parser.add_mutually_exclusive_group()
    foldergroup.add_argument(
        "-layout",
        choices=("claude", "agents"),
        default="claude",
        help=(
            "The skills folder of the agent. claude gives ~/.claude/skills, which Claude Code reads. If"
            " CLAUDE_CONFIG_DIR has a value, claude gives $CLAUDE_CONFIG_DIR/skills. agents gives ~/.agents/skills,"
            " which Codex, Cursor, and OpenCode read"
        ),
    )
    foldergroup.add_argument(
        "-skillsdir", type=Path, help="A skills folder of the user in place of the folder that -layout gives"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Write a copy of the skill and not a link. Install the copy again after an update of artistools",
    )
    addarg_force(parser)


def install_skill(skillsdir: Path, *, copy: bool, force: bool) -> bool:
    """Put the skill into the skills folder. Return False when the link to the package is there already."""
    source = get_skill_folder().resolve()
    destination = skillsdir / SKILLNAME
    islinktosource = destination.is_symlink() and destination.exists() and destination.samefile(source)
    if islinktosource and not copy:
        return False
    # samefile also finds the folder through a different spelling, e.g. other capital letters on macOS
    if not islinktosource and is_in_folder(destination.resolve(), source):
        exit_with_error(
            f"{destination} is the folder of the skill in the package, or is in it", "Give a different -skillsdir"
        )
    # rmtree of a real folder that holds the package deletes the skill of the package, and maybe artistools too. The
    # command removes a link and not its target, thus only a real folder needs this test
    if destination.is_dir() and not destination.is_symlink() and is_in_folder(source, destination.resolve()):
        exit_with_error(
            f"{destination} holds the skill of the package, e.g. in artistools", "Give a different -skillsdir"
        )
    if not copy and is_in_uv_cache(source):
        exit_with_error(
            f"This installation of artistools runs from the uv cache, which uv can delete. Thus a link to {source} can"
            " break",
            "Install artistools with `uv tool install artistools`, or give --copy",
        )

    # is_symlink also finds a link to a folder that does not exist
    if destination.exists() or destination.is_symlink():
        if not force and not (destination / "SKILL.md").is_file():
            exit_with_error(f"{destination} exists, and it holds no SKILL.md", "Give -f or --force to replace it")

        # the command removes a link and not the folder of the link, but rmtree deletes each file of a real folder
        filenames = (
            ["SKILL.md"]
            if destination.is_symlink() or not destination.is_dir()
            else sorted(str(path.relative_to(destination)) for path in destination.rglob("*") if path.is_file())
        )
        confirm_overwrite([destination], filenames, force)
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        else:
            shutil.rmtree(destination)

    skillsdir.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copytree(source, destination)
    else:
        destination.symlink_to(source, target_is_directory=True)
    return True


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Install the artis-data skill for a large language model (LLM) agent, e.g. Claude Code."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)
    # a shell does not expand the ~ of "-skillsdir=~/skills" or of a quoted path
    skillsdir = args.skillsdir.expanduser() if args.skillsdir else get_layout_skillsdir(args.layout)
    destination = skillsdir / SKILLNAME
    if install_skill(skillsdir, copy=args.copy, force=args.force):
        print(f"Installed {SKILLNAME} as a {'copy' if args.copy else 'link'} in {destination}")
    else:
        print(f"{destination} is already a link to the skill of this installation of artistools")
    print_detail("Start a new session of the agent. Then ask a question about a model.")
    if args.skillsdir is None:
        print_detail(
            f"In Claude Code, you can also type /{SKILLNAME}."
            if args.layout == "claude"
            else f"In Codex, you can also type ${SKILLNAME}."
        )
