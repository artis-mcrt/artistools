"""Print the tab-completion code of every artistools command for a shell.

Put this line in the startup file of the shell, e.g. ~/.zshrc after compinit, or ~/.bashrc:

    eval "$(artistools completions zsh)"

The code asks artistools for the completions at each press of the Tab key, thus a new argument needs
no other step. Only a new console script needs the code again.
"""

import argparse
import os
import typing as t
from collections.abc import Sequence
from pathlib import Path

from artistools.misc import parse_cli_args

SHELLS = ("bash", "zsh", "fish", "tcsh", "powershell")


def get_default_shell() -> str:
    """Return the name of the login shell of the user, or bash if argcomplete does not support it."""
    shell = Path(os.environ.get("SHELL", "")).name
    return shell if shell in SHELLS else "bash"


def get_completion_code(shell: str) -> str:
    """Return the code that registers the completions of the dispatcher and of each console script."""
    import argcomplete

    from artistools.commands import DISPATCHERSCRIPTS
    from artistools.commands import get_script_subcommands

    return argcomplete.shellcode([*DISPATCHERSCRIPTS, *sorted(get_script_subcommands())], shell=shell)


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add the argument that names the shell."""
    parser.add_argument(
        "shell",
        nargs="?",
        choices=SHELLS,
        default=get_default_shell(),
        help="Shell to write the code for. The default comes from the SHELL environment variable",
    )


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Print the tab-completion code for a shell. Use it as: eval "$(artistools completions zsh)"."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)
    print(get_completion_code(args.shell))
