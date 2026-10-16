"""Install the artis-data skill. It tells a large language model (LLM) agent how to read the data of a model.

The command makes a link to the skill in the installed package. An update of artistools then also updates the
skill. With --copy, the command writes a copy. Install the skill again after each update of artistools.
"""

import argparse
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


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add the arguments that select the skills folder and the type of the installation."""
    parser.add_argument(
        "-skillsdir",
        type=Path,
        default=Path.home() / ".claude" / "skills",
        help="The folder that holds the skills of the user. The default is ~/.claude/skills, which Claude Code reads",
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
    if destination.resolve() == source:
        if destination.is_symlink() and not copy:
            return False
        if not destination.is_symlink():
            exit_with_error(f"{destination} is the folder of the skill in the package", "Give a different -skillsdir")

    # is_symlink also finds a link to a folder that does not exist
    if destination.exists() or destination.is_symlink():
        if (destination / "SKILL.md").is_file():
            confirm_overwrite([destination], ["SKILL.md"], force)
        elif not force:
            exit_with_error(f"{destination} exists, and it holds no SKILL.md", "Give -f or --force to replace it")

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
    destination = args.skillsdir / SKILLNAME
    if install_skill(args.skillsdir, copy=args.copy, force=args.force):
        print(f"Installed {SKILLNAME} as a {'copy' if args.copy else 'link'} in {destination}")
    else:
        print(f"{destination} is already a link to the skill of this installation of artistools")
    print_detail("Start a new session of the agent. Then ask a question about a model, or type /artis-data.")
