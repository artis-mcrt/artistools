"""Sphinx configuration for the artistools documentation."""

import os
import sys
import typing as t
from importlib.metadata import version as get_version
from pathlib import Path

from artistools.commands import COMMANDGROUPS
from artistools.commands import CommandSpec
from artistools.commands import GROUPHELPTEXT
from artistools.commands import subcommandtree

if t.TYPE_CHECKING:
    from collections.abc import Generator

    from artistools.commands import CommandTree

project = "artistools"
author = "Luke J. Shingles and the ARTIS Collaboration"
project_copyright = "2017-%Y, the ARTIS Collaboration"
release = get_version("artistools")
version = release

extensions = ["myst_parser", "sphinx_copybutton", "sphinxcontrib.programoutput"]

# All pages are MyST Markdown. Sphinx reads no docstring, thus the documentation contains no reStructuredText.
source_suffix = {".md": "markdown"}
exclude_patterns = ["_build"]

myst_heading_anchors = 3

# The canonical link of a page names the version that a search engine must show.
# A search engine then shows that version alone.
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")

# The "Edit this page" link of each page points at the branch or the tag that Read the Docs builds.
# A build of a pull request gives the number of the request, which is not a branch.
# The link then points at the main branch.
source_branch = os.environ.get("READTHEDOCS_GIT_IDENTIFIER", "main")
if os.environ.get("READTHEDOCS_VERSION_TYPE") == "external":
    source_branch = "main"

html_theme = "furo"
html_title = f"artistools {release}"
html_theme_options = {
    "source_repository": "https://github.com/artis-mcrt/artistools/",
    "source_branch": source_branch,
    "source_directory": "docs/",
}

# The program-output directive starts the console scripts.
# The Python that runs Sphinx can be in a virtual environment that is not on PATH.
# argparse wraps the help text to the terminal width.
os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
os.environ["COLUMNS"] = "80"


def iter_visible_commands(name: str, node: "CommandSpec | CommandTree") -> "Generator[tuple[str, CommandSpec]]":
    """Yield (full name, spec) for each command that the help lists at or below one node of the tree."""
    if isinstance(node, CommandSpec):
        if not node.hidden:
            yield name, node
    else:
        for subname, subnode in node.items():
            yield from iter_visible_commands(f"{name} {subname}", subnode)


def get_command_section(name: str, spec: CommandSpec) -> list[str]:
    """Return the Markdown lines of the section of one command."""
    lines = [f"### {name}", "", spec.get_description(), ""]
    if spec.examples:
        examples = [f"artistools {name} {arguments}".rstrip() + f"  # {what}" for arguments, what in spec.examples]
        lines += ["```sh", *examples, "```", ""]
    return [*lines, f"```{{program-output}} artistools {name} --help", "```", ""]


def get_commands_page() -> str:
    """Return the Markdown text of the page that lists each command of subcommandtree."""
    lines = [
        "# Commands",
        "",
        "Each command is a subcommand of `artistools`, and `at` is a short name for `artistools`.",
        "The build of this page reads the list and the help text from the code.",
        "",
    ]
    # --help lists a command under the last heading if no tuple of COMMANDGROUPS names that command
    lastheading = list(COMMANDGROUPS)[-1]
    groupofcommand = {name: heading for heading, names in COMMANDGROUPS.items() for name in names}
    for heading in COMMANDGROUPS:
        members = [name for name in subcommandtree if groupofcommand.get(name, lastheading) == heading]
        commands = [command for name in members for command in iter_visible_commands(name, subcommandtree[name])]
        if not commands:
            continue

        lines += [f"## {heading}", ""]
        lines += [GROUPHELPTEXT[name] + "\n" for name in members if name in GROUPHELPTEXT]
        lines += ["| Command | Other names | Description |", "| --- | --- | --- |"]
        for name, spec in commands:
            anchor = name.replace(" ", "-")
            lines.append(f"| [{name}](#{anchor}) | {', '.join(spec.aliases)} | {spec.helptext} |")

        lines.append("")
        for name, spec in commands:
            lines += get_command_section(name, spec)

    return "\n".join(lines)


def write_commands_page() -> None:
    """Write commands.md only if the text is different.

    Sphinx reads the file again after each write.
    """
    page = Path(__file__).parent / "commands.md"
    text = get_commands_page()
    if not page.is_file() or page.read_text(encoding="utf-8") != text:
        page.write_text(text, encoding="utf-8")


write_commands_page()
