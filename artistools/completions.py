"""Give the instructions for tab completion, or print the tab-completion code for a shell.

With no shell, the command writes the instructions for the shell in the SHELL environment variable to the
standard error. With a shell, it prints the code, e.g. artistools completions zsh > ~/.zfunc/_artistools.

The code asks artistools for the completions at each press of the Tab key, thus a new argument needs
no other step. Only a new console script needs the file again.
"""

import argparse
import os
import sys
import typing as t
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType

from artistools.misc import parse_cli_args
from artistools.misc import print_product

# zsh and fish load a file from a known folder at each start. bash and PowerShell need one line in the
# startup file that reads the file
INSTRUCTIONS: Mapping[str, str] = MappingProxyType({
    "zsh": """\
Run these commands:
    mkdir -p ~/.zfunc
    artistools completions zsh > ~/.zfunc/_artistools

Then add this line to ~/.zshrc before the line that runs compinit:
    fpath=(~/.zfunc $fpath)
zsh then loads the file at each start. The line must put ~/.zfunc first, because zsh has a
completion for a different at command, and compinit keeps the first one that it finds.
If the Tab key gives no completions in a new shell, delete ~/.zcompdump. Then start a new shell.""",
    "bash": """\
Run this command:
    artistools completions bash > ~/.artistools-completion.bash

Then add this line to ~/.bashrc:
    source ~/.artistools-completion.bash""",
    "fish": """\
Run this command:
    artistools completions fish > ~/.config/fish/conf.d/artistools.fish

fish loads the file at the next start.""",
    "powershell": """\
Run this command:
    artistools completions powershell > ~/artistools-completion.ps1

Then add this line to $PROFILE:
    . ~/artistools-completion.ps1""",
})


def get_completion_code(shell: str) -> str:
    """Return the code that registers the completions of the dispatcher and of each console script."""
    import argcomplete

    from artistools.commands import DISPATCHERSCRIPTS
    from artistools.commands import get_script_subcommands

    return argcomplete.shellcode([*DISPATCHERSCRIPTS, *sorted(get_script_subcommands())], shell=shell)


def get_instructions(shell: str) -> str:
    """Return the instructions to enable tab completion in a shell, or in every shell for an unknown name."""
    shells = [shell] if shell in INSTRUCTIONS else list(INSTRUCTIONS)
    lines = [line for name in shells for line in (f"To enable tab completion in {name}:", "", INSTRUCTIONS[name], "")]
    lines.append("Write the file again after an update of artistools that adds a console script.")
    if shell in INSTRUCTIONS:
        othershells = [othershell for othershell in INSTRUCTIONS if othershell != shell]
        lines += [
            f"For a different shell ({', '.join(othershells)}), give its name in SHELL, e.g.:",
            f"    SHELL={othershells[0]} artistools completions",
        ]
    return "\n".join(lines)


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add the argument that names the shell."""
    parser.add_argument(
        "shell",
        nargs="?",
        choices=tuple(INSTRUCTIONS),
        help="Shell to print the code for. With no shell, the command gives the instructions",
    )


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Give the instructions for tab completion, or print the code for the given shell."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)
    if args.shell:
        print_product(args, get_completion_code(args.shell))
    else:
        # a redirect of the old command, "artistools completions > file", must not write this text to a file
        # that the shell reads, thus the instructions go to the standard error
        print(get_instructions(Path(os.environ.get("SHELL", "")).name), file=sys.stderr)
