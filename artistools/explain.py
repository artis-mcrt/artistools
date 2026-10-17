"""Explain an ARTIS quantity, e.g. TJ or gamma_NT_Fe_II: the meaning, the unit, the method, and the papers.

The text comes from the files of the artis-data skill, thus this command and the skill always agree. SKILL.md and
quantities.md give the meaning and the unit of each column. physics.md gives the method and the papers.
"""

import argparse
import re
import textwrap
import typing as t
from collections.abc import Sequence
from functools import lru_cache

from artistools.estimators import get_units
from artistools.estimators import scan_estimators
from artistools.installskill import get_skill_folder
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_positional_items
from artistools.misc import exit_with_error
from artistools.misc import parse_cli_args
from artistools.misc import print_product
from artistools.misc import resolve_positional_modelpath
from artistools.misc import suggest_names

# SKILL.md gives the main columns of each reader, and quantities.md gives the other columns
ENTRYFILES = ("SKILL.md", "quantities.md")
PHYSICSFILE = "physics.md"

LISTVARIABLESHINT = "'artistools plotestimators --listvariables <modelpath>' lists the variables of a model."

# a name in backticks, e.g. `TJ`, `luminosity_erg/s`, or `nnion_<El>_<ion stage>`. A function call, a command, or a
# file name has a dot, a bracket, or a space outside a placeholder, thus this pattern does not take it
NAMEPATTERN = r"[A-Za-z][\w/*]*(?:<[\w ]+>[\w/*]*)*"
HEADNAME = re.compile(rf"`(?P<name>{NAMEPATTERN})`(?: \((?P<bracket>[^()]*(?:\([^()]*\)[^()]*)*)\))?")
HEADSEPARATOR = re.compile(r",? (?:and|or) |, (?:or )?")
# the text after the names of an entry defines them, e.g. ", the energy", ": the part", "is the", or "of each cell"
DEFINITIONSTART = re.compile(r"[,:;.]|is |are |of |$")
UNIT = re.compile(r"[A-Za-zÅ]+(?:\^-?\d+)?(?:/[A-Za-zÅ]+(?:\^-?\d+)?)*")
# a reference starts with the authors and the year, e.g. "Kromer M., Sim S. A., 2009, MNRAS"
REFERENCESTART = re.compile(r"(\S+) .*?, ((?:19|20)\d\d)[a-z]?,")
# the units that the end of a name gives, e.g. lambda_angstroms. A longer suffix comes first
NAMEUNITSUFFIXES = {
    "_erg/s/g": "erg/s/g",
    "_erg/s": "erg/s",
    "_Lsun": "Lsun",
    "_angstroms": "Å",
    "_days": "days",
    "_g": "g",
}
# "e.g." and "i.e." do not end a sentence
SENTENCEEND = re.compile(r"(?<!e\.g\.)(?<!i\.e\.)(?<=[.!?])\s+(?=[A-Z*])")

# the values that a placeholder of an entry takes. Another placeholder, e.g. <process>, takes a word
PLACEHOLDERPATTERNS = {
    "El": r"[A-Z][a-z]?",
    "ion stage": r"[IVXL]+",
    "ion": r"[A-Z][a-z]?_[IVXL]+",
    "nuclide": r"[A-Z][a-z]?\d+",
}
WORDPATTERN = r"[^\W_]\w*"


class SkillBlock(t.NamedTuple):
    """One paragraph or one list item of a skill file."""

    filename: str
    heading: str
    text: str
    leadin: str
    """The paragraph before the list of a list item, or an empty string."""


class QuantityEntry(t.NamedTuple):
    """The text that a skill file gives for one name, e.g. gamma_NT_<ion>."""

    name: str
    label: str
    """The meaning in the brackets after the name, e.g. "electron temperature"."""

    unit: str
    description: str
    """The text after the names, which can define more than one name."""

    block: SkillBlock


def strip_markup(text: str) -> str:
    """Remove the bold marks of markdown."""
    return text.replace("**", "")


@lru_cache(maxsize=8)
def get_skill_blocks(filename: str) -> tuple[SkillBlock, ...]:
    """Return the paragraphs and the list items of a file of the artis-data skill, in the order of the file."""
    blocks: list[SkillBlock] = []
    heading = ""
    leadin = ""
    lines: list[str] = []
    isitem = False
    incode = False

    def add_block() -> None:
        nonlocal leadin
        if lines:
            text = " ".join(lines)
            blocks.append(SkillBlock(filename, heading, text, leadin if isitem else ""))
            if not isitem:
                leadin = text
        lines.clear()

    for line in (get_skill_folder() / filename).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            add_block()
            incode = not incode
        elif incode:
            continue
        elif stripped.startswith("#"):
            add_block()
            heading = stripped.lstrip("# ").replace("`", "")
            leadin = ""
        elif not stripped:
            add_block()
            isitem = False
        elif stripped.startswith("- "):
            add_block()
            isitem = True
            lines.append(stripped.removeprefix("- "))
        else:
            lines.append(stripped)
    add_block()

    return tuple(blocks)


def split_bracket(bracket: str) -> tuple[str, str]:
    """Return the meaning and the unit in the text of a bracket, e.g. "electron temperature, K" or "erg/s/cm^3"."""
    meaning, _, lastpart = bracket.rpartition(", ")
    if UNIT.fullmatch(lastpart):
        return meaning, lastpart
    return bracket, ""


def get_entry_candidates(filename: str) -> list[SkillBlock]:
    """Return the list items of a skill file that start with a name in backticks."""
    return [block for block in get_skill_blocks(filename) if block.leadin and HEADNAME.match(block.text)]


def parse_entry(block: SkillBlock) -> list[QuantityEntry]:
    """Return one entry for each name at the start of a list item, or an empty list when the item defines no name.

    An item such as "`get_spectra` takes time steps" does not define its name, thus it gives no entry.
    """
    heads: list[tuple[str, str]] = []
    position = 0
    while match := HEADNAME.match(block.text, position):
        heads.append((match["name"], match["bracket"] or ""))
        position = match.end()
        separator = HEADSEPARATOR.match(block.text, position)
        if separator is None or not HEADNAME.match(block.text, separator.end()):
            break
        position = separator.end()

    rest = block.text[position:].lstrip()
    if not heads or (not any(bracket for _, bracket in heads) and not DEFINITIONSTART.match(rest)):
        return []

    description = re.sub(r"^(?:[,:;.] *|is |are )", "", rest).rstrip(";.")
    # a name with no bracket takes the bracket of the next name, e.g. `em_posx`, `em_posy`, `em_posz` (cm). After the
    # last bracket, a unit can follow in the first clause, e.g. `heating_<process>` of each cell (erg/s/cm^3):
    firstclause = re.split(r": |\. |; ", rest, maxsplit=1)[0]
    clauseunits = [unit for unit in re.findall(r"\(([^()]*)\)", firstclause) if UNIT.fullmatch(unit)]
    nextbracket = ""
    entries: list[QuantityEntry] = []
    for name, bracket in reversed(heads):
        nextbracket = bracket or nextbracket
        label, groupunit = split_bracket(nextbracket) if nextbracket else ("", clauseunits[0] if clauseunits else "")
        ownunit = split_bracket(bracket)[1]
        nameunit = next((unit for suffix, unit in NAMEUNITSUFFIXES.items() if name.endswith(suffix)), "")
        entries.append(QuantityEntry(name, label, ownunit or nameunit or groupunit, description, block))

    return entries[::-1]


@lru_cache(maxsize=1)
def get_entries() -> tuple[QuantityEntry, ...]:
    """Return the entries of all the names that the skill files define."""
    return tuple(
        entry for filename in ENTRYFILES for block in get_entry_candidates(filename) for entry in parse_entry(block)
    )


@lru_cache(maxsize=256)
def get_name_pattern(name: str) -> re.Pattern[str]:
    """Return a pattern that matches each column of a name, e.g. gamma_NT_Fe_II for gamma_NT_<ion>.

    A name that ends with an underscore, e.g. cooling_, names the start of a family.
    """
    parts = re.split(r"<([\w ]+)>", name)
    pattern = "".join(
        re.escape(part) if index % 2 == 0 else f"(?P<p{index // 2}>{PLACEHOLDERPATTERNS.get(part, WORDPATTERN)})"
        for index, part in enumerate(parts)
    )
    return re.compile(pattern + (WORDPATTERN if name.endswith("_") else ""))


def get_literal_length(name: str) -> int:
    """Return the number of characters of a name outside its placeholders. A longer literal part is more specific."""
    return len(re.sub(r"<[\w ]+>", "", name))


def find_entries(name: str) -> list[QuantityEntry]:
    """Return the most specific entries of a name, e.g. heating_<process>_<ion> and not heating_<process>."""
    matches = [entry for entry in get_entries() if get_name_pattern(entry.name).fullmatch(name)]
    if not matches:
        return []
    maxlength = max(get_literal_length(entry.name) for entry in matches)
    return [entry for entry in matches if get_literal_length(entry.name) == maxlength]


def find_physics_blocks(name: str) -> list[SkillBlock]:
    """Return the blocks of physics.md that name the quantity in backticks, e.g. `TJ` or `gamma_NT_<ion>`."""
    return [
        block
        for block in get_skill_blocks(PHYSICSFILE)
        if any(get_name_pattern(token).fullmatch(name) for token in re.findall(rf"`({NAMEPATTERN})`", block.text))
    ]


@lru_cache(maxsize=1)
def get_references() -> tuple[tuple[str, str, str], ...]:
    """Return the first author, the year, and the full reference of each paper in physics.md."""
    return tuple(
        (match[1], match[2], block.text)
        for block in get_skill_blocks(PHYSICSFILE)
        if block.heading == "References" and (match := REFERENCESTART.match(block.text))
    )


def get_cited_references(text: str) -> list[str]:
    """Return the full reference of each paper that the text cites, e.g. as "Shingles et al. (2020)"."""
    # a citation does not continue past a semicolon or a closing bracket, e.g. "(Sim 2007; Kromer & Sim 2009)"
    return [
        reference
        for author, year, reference in get_references()
        if re.search(rf"\b{re.escape(author)}\b[^;)]{{0,40}}?\b{year}\b", text)
    ]


def get_first_sentences(text: str, count: int) -> str:
    """Return the first sentences of a text, with a full stop at the end, or an empty string."""
    sentences = " ".join(SENTENCEEND.split(strip_markup(text).strip().rstrip(";:."))[:count])
    return f"{sentences.rstrip('.')}." if sentences else ""


def describe_unit(entry: QuantityEntry, name: str) -> str:
    """Return the unit of an entry, and the unit that artistools gives for an estimator when it is different."""
    estimatorunit = get_units(name, latex=False) if "estimators" in entry.block.heading.lower() else None
    if estimatorunit is None or estimatorunit == entry.unit:
        return entry.unit
    if not entry.unit:
        return f"{estimatorunit} (from artistools, the skill gives no unit)"
    return f"{entry.unit} (artistools gives {estimatorunit})"


def describe_placeholders(entry: QuantityEntry, name: str) -> list[str]:
    """Return a line for each placeholder value that the description explains, e.g. "process ff: free-free"."""
    match = get_name_pattern(entry.name).fullmatch(name)
    if match is None:
        return []
    placeholders = re.findall(r"<([\w ]+)>", entry.name)
    lines = []
    for index, placeholder in enumerate(placeholders):
        value = match[f"p{index}"]
        if explanation := re.search(rf"`{re.escape(value)}` \(([^()]+)\)", entry.description):
            lines.append(f"{placeholder} {value}: {explanation[1]}")
    return lines


def wrap(text: str, indent: str) -> str:
    """Wrap a line of the explanation, and indent the continuation lines below the text."""
    return textwrap.fill(
        text,
        width=120,
        initial_indent=indent,
        subsequent_indent=indent + "  ",
        break_long_words=False,
        break_on_hyphens=False,
    )


def explain_name(name: str) -> str:
    """Return the explanation of one name that has an entry."""
    lines = [name]
    for entry in find_entries(name):
        source = f"{entry.block.filename}, section '{entry.block.heading}'"
        lines.append(wrap(f"Entry: {entry.name} in {source}", "  "))
        description = get_first_sentences(entry.description, 2)
        if description.startswith("of "):
            description = f"`{entry.name}` {description}"
        meaning = entry.label or description
        lines.append(wrap(f"Meaning: {meaning or 'the entry gives no meaning.'}", "    "))
        if entry.label and description:
            lines.append(wrap(f"Details: {description}", "    "))
        lines.extend(wrap(line, "    ") for line in describe_placeholders(entry, name))
        if unit := describe_unit(entry, name):
            lines.append(wrap(f"Unit: {unit}", "    "))

    physicsblocks = find_physics_blocks(name)
    if not physicsblocks:
        lines.append(f"  Method: {PHYSICSFILE} does not describe this quantity.")
    papers: dict[str, None] = {}
    for block in physicsblocks:
        method = get_first_sentences(block.text, 2)
        if block.leadin:
            method = f"{get_first_sentences(block.leadin, 1)} {method}"
        lines.append(wrap(f"Method ({PHYSICSFILE}, section '{block.heading}'): {method}", "  "))
        papers |= dict.fromkeys(get_cited_references(f"{block.leadin} {block.text}"))
    if papers:
        lines.append("  Papers:")
        lines.extend(wrap(f"- {paper}", "    ") for paper in papers)

    return "\n".join(lines)


def get_unknown_name_help(name: str, modelcolumns: Sequence[str] | None, modelpathtext: str) -> str:
    """Return the help text for a name that has no entry: the closest names and the columns of the model."""
    candidates = [entry.name for entry in get_entries()] + list(modelcolumns or [])
    parts = [suggest_names(name, dict.fromkeys(candidate for candidate in candidates if candidate != name))]
    if modelcolumns is not None:
        has = "has the" if name in modelcolumns else "has no"
        parts.append(f"The model {modelpathtext} {has} estimator column '{name}'.")
    text = " ".join(part for part in parts if part)
    return f"'{name}': {text}" if text else ""


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_positional_items(
        parser,
        dest="names",
        metavar="name",
        helptext=(
            "The names of the quantities, e.g. 'artistools explain TJ gamma_NT_Fe_II'. A model folder can come after"
            " them. For a name with no explanation, the command then says if the estimators of that model have it"
        ),
    )

    # the default is None and not Path(), thus resolve_positional_modelpath sees an explicit value
    addarg_modelpath(parser)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Explain an ARTIS quantity: the meaning, the unit, the method, and the papers."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)
    givennamecount = len(args.names)
    modelpathgiven = args.modelpath is not None
    items = resolve_positional_modelpath(args, "names")
    # resolve_positional_modelpath removes a model folder from the end of the items
    modelpathgiven = modelpathgiven or len(items) < givennamecount
    names = list(dict.fromkeys(items))
    if not names:
        exit_with_error("no name was given", "Give the name of a quantity, e.g. artistools explain TJ")

    if unknownnames := [name for name in names if not find_entries(name)]:
        modelcolumns = scan_estimators(args.modelpath).collect_schema().names() if modelpathgiven else None
        helptexts = [get_unknown_name_help(name, modelcolumns, str(args.modelpath)) for name in unknownnames]
        exit_with_error(
            f"the artis-data skill does not explain {', '.join(repr(name) for name in unknownnames)}",
            " ".join([*helptexts, LISTVARIABLESHINT]).strip(),
        )

    print_product(args, "\n\n".join(explain_name(name) for name in names))
