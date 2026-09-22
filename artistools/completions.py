"""Give the instructions for tab completion, or print the tab-completion code for a shell.

With no shell, the command prints the line to put in the startup file of each shell. With a shell, it
prints the code that the line reads, e.g. eval "$(artistools completions zsh)" in ~/.zshrc.

The code asks artistools for the completions at each press of the Tab key, thus a new argument needs
no other step. Only a new console script needs the code again.
"""

import argparse
import os
import typing as t
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType

from artistools.misc import parse_cli_args

# the startup file of each shell, and the line in it that reads the code
STARTUPLINES: Mapping[str, tuple[str, str]] = MappingProxyType({
    "zsh": ("~/.zshrc, after the line that runs compinit", 'eval "$(artistools completions zsh)"'),
    "bash": ("~/.bashrc", 'eval "$(artistools completions bash)"'),
    "fish": ("~/.config/fish/config.fish", "artistools completions fish | source"),
    "tcsh": ("~/.tcshrc", "eval `artistools completions tcsh`"),
    "powershell": ("$PROFILE", "artistools completions powershell | Out-String | Invoke-Expression"),
})


def get_completion_code(shell: str) -> str:
    """Return the code that registers the completions of the dispatcher and of each console script."""
    import argcomplete

    from artistools.commands import DISPATCHERSCRIPTS
    from artistools.commands import get_script_subcommands

    return argcomplete.shellcode([*DISPATCHERSCRIPTS, *sorted(get_script_subcommands())], shell=shell)


def get_instructions() -> str:
    """Return the instructions to enable tab completion, with the shell of the user first."""
    usershell = Path(os.environ.get("SHELL", "")).name
    shells = sorted(STARTUPLINES, key=lambda shell: shell != usershell)
    lines = ["To enable tab completion, add one line to the startup file of your shell.", ""]
    for shell in shells:
        startupfile, startupline = STARTUPLINES[shell]
        lines += [
            f"{shell}{' (your shell)' if shell == usershell else ''}: in {startupfile}, add",
            f"    {startupline}",
            "",
        ]
    lines += [
        "The eval line runs artistools at each start of the shell. For a faster start, write the code to a",
        "file one time, and source that file in place of the line. Write the file again after an update that",
        "adds a console script, e.g.:",
        f"    artistools completions {shells[0]} > ~/.artistools-completion.{shells[0]}",
    ]
    return "\n".join(lines)


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add the argument that names the shell."""
    parser.add_argument(
        "shell",
        nargs="?",
        choices=tuple(STARTUPLINES),
        help="Shell to print the code for. With no shell, the command prints the instructions",
    )


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Give the instructions for tab completion, or print the code for the given shell."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)
    print(get_completion_code(args.shell) if args.shell else get_instructions())
