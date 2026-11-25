"""Read and write the plot items of plotestimators, e.g. the variables, the ions, the directives, and the styles.

The command and its viewer both use this module, thus the viewer reads each subplot as the command does. A subplot of
the viewer holds the --plot tokens of the command, e.g. ("populations", "Fe II@color=C3", "yscale=log").
"""

import argparse
import contextlib
import math
import re
import typing as t
from collections.abc import Collection
from collections.abc import Mapping
from collections.abc import Sequence
from types import MappingProxyType

import numpy as np
import polars as pl

from artistools.atomic import decode_roman_numeral
from artistools.atomic import get_atomic_number
from artistools.atomic import get_elsymbol
from artistools.atomic import get_elsymbolset
from artistools.atomic import get_ion_tuple
from artistools.atomic import get_ionstring
from artistools.estimators.core import VARIABLES
from artistools.misc import color_arg
from artistools.misc import dashes_arg
from artistools.misc import exit_with_error
from artistools.misc import LINESTYLE_ALIASES
from artistools.misc import LINESTYLE_NAMES
from artistools.misc import positive_float_arg
from artistools.misc import SERIES_STYLE_KEYS
from artistools.misc import suggest_names

VARIABLE_ALIASES: t.Final[Mapping[str, str]] = MappingProxyType({"T_e": "Te", "n_e": "nne", "T_R": "TR", "T_J": "TJ"})


# a series type groups the names that follow it, e.g. -plot averageionisation Fe Ni. Each one has its
# own function in get_subplot_plans, e.g. get_average_ionisation_plans. An ion series needs no entry here,
# because the estimator columns name it, e.g. gamma_NT_Fe_II gives the series type gamma_NT
SERIESTYPES = (
    "averageexcitation",
    "averageionisation",
    "initabundances",
    "initmasses",
    "levelpopulation",
    "populations",
)


def is_seriestype(name: t.Any, estimatorcolumns: Collection[str]) -> bool:
    """Return True when a name has its own function of series plans, e.g. "populations" in "populations Fe II"."""
    if not isinstance(name, str) or name in estimatorcolumns:
        return False

    return name in SERIESTYPES or name.startswith("levelpopulation_")


def get_iontuple(ionstr: str) -> tuple[int, str | int]:
    """Decode into atomic number and parameter, e.g., [(26, 1), (26, 2), (26, 'ALL'), (26, 'Fe56')].

    An element gives "ALL". An isotope or a family of nuclides keeps its name, e.g. Fe56, Ni_56, or
    Ni_stable, because the column name of the estimators holds that name. Every other name goes to
    get_ion_tuple.
    """
    elsymbols = get_elsymbolset()
    if ionstr not in elsymbols and not ionstr.isdigit():
        stem = ionstr.rstrip("-0123456789")
        if stem in elsymbols:
            return (get_atomic_number(stem), ionstr)

        elsymbol, _, suffix = ionstr.partition("_")
        if suffix and elsymbol in elsymbols and decode_roman_numeral(suffix) < 0:
            return (get_atomic_number(elsymbol), ionstr)

    # get_ion_tuple strips an X_, nnelement_, or nnion_ prefix. Such a prefix names a column and not
    # an ion, thus a name that holds one goes to the fallback below and the caller then rejects it
    if not ionstr.startswith(("X_", "nnelement_", "nnion_")):
        with contextlib.suppress(ValueError):
            iontuple = get_ion_tuple(ionstr)
            return (iontuple, "ALL") if isinstance(iontuple, int) else iontuple

    # a name that is no ion at all, e.g. a mistyped variable. The caller tests the atomic number
    return (get_atomic_number(ionstr.split("_", maxsplit=1)[0]), ionstr)


def is_valid_ion(ionstr: str) -> bool:
    """Return True when the string names an element, an ion, or an isotope that get_iontuple can read.

    could_be_ion is deliberately permissive, thus it accepts a name such as "te" that get_atomic_number
    reads as tellurium. This test rejects that name, so that the caller can suggest the variable "Te".
    """
    atomic_number, param = get_iontuple(ionstr)
    if atomic_number < 1:
        return False

    if isinstance(param, int):
        return param >= 1

    elsymbols = get_elsymbolset()

    # get_column_name reads a suffix that joins the symbol, e.g. Fe_otherstable gives
    # nniso_Fe_otherstable, thus the first part of the name decides
    return param == "ALL" or param.rstrip("-0123456789") in elsymbols or param.split("_", maxsplit=1)[0] in elsymbols


def could_be_ion(plotvar: t.Any) -> bool:
    """Return True when plotvar can name part of an ion population plot: an ion, an element, or an atomic number."""
    # lists are plot directives and bare integers are atomic numbers
    if isinstance(plotvar, (list, int)):
        return True

    if not isinstance(plotvar, str) or is_expression_item(plotvar):
        return False

    # get_iontuple reads a digit string as an atomic number, and normalise_plotitems has already taken
    # every "key=value" directive out of the list that reaches here
    return get_iontuple(plotvar)[0] >= 1


def get_iontuple_sortkey(iontuple: tuple[int, str | int]) -> tuple[int, int, int, str]:
    """Return a sort key that puts an element first, then its ion stages, then its isotopes.

    A plain sort of the tuples compares an ion stage with the name of an isotope, which raises a
    TypeError, e.g. for the pair of names "Fe" and "Fe II".
    """
    atomic_number, ion_stage = iontuple
    if ion_stage == "ALL":
        return (atomic_number, 0, 0, "")

    if isinstance(ion_stage, int):
        return (atomic_number, 1, ion_stage, "")

    return (atomic_number, 2, 0, ion_stage)


def get_column_name(seriestype: str, atomic_number: int, ion_stage: str | int) -> tuple[str, str]:
    """Return the estimator column name for one ion, element, or isotope, along with its plot label."""
    ionstr = get_ionstring(atomic_number, ion_stage, sep="_", style="spectral")
    if seriestype == "populations":
        if ion_stage == "ALL":
            elsymbol = get_elsymbol(atomic_number)
            return f"nnelement_{elsymbol}", ionstr
        if isinstance(ion_stage, str) and ion_stage.startswith(get_elsymbol(atomic_number)):
            # not really an ion_stage but an isotope name
            return f"nniso_{ion_stage}", ionstr
        return f"nnion_{ionstr}", ionstr
    return f"{seriestype}_{ionstr}", ionstr


def is_ionseriestype(name: t.Any, estimatorcolumns: Collection[str], params: Sequence[t.Any]) -> bool:
    """Return True when a name plus the ions after it give the columns of an ion series.

    An estimator that names an ion, e.g. gamma_NT_Fe_II, gives the series type gamma_NT. Every name
    after it must be an ion, because a name such as "heating" is also the prefix of heating_coll, and
    "heating coll" must keep the message that names the column.

    A type of series can also be a variable of its own. cooling_coll holds the cooling rate of the
    whole cell, and cooling_coll_Fe_II holds the part of one ion. A column of one of the named ions
    then proves the reading. A subplot of the total alone takes its own --plot, e.g.
    --plot cooling_coll --plot cooling_coll "Fe II".
    """
    if not isinstance(name, str) or not params:
        return False

    if not all(isinstance(param, str) and is_valid_ion(param) for param in params):
        return False

    # A column of one of the ions proves the reading, also for a name that is a variable of its own.
    # The test takes one column and not every column. An element can lose its top ion here, e.g. a
    # model that holds Fe I to Fe V has no gamma_NT_Fe_V.
    if any(get_column_name(name, *get_iontuple(param))[0] in estimatorcolumns for param in params):
        return True

    # A name that gives no variable of its own keeps the older and looser test, which asks only that
    # the model holds the family. Such a name has no other reading, thus it needs no column here.
    return name not in estimatorcolumns and any(col.startswith(f"{name}_") for col in estimatorcolumns)


# The plot directives that a plot item can carry, e.g. "-p rho yscale=log". The underscore of a name
# is optional.
DIRECTIVES = ("ionpoptype", "ymin", "ymax", "yscale")


# The subplots share one horizontal axis, thus no directive can set it for one subplot alone. The
# arguments -xmin and -xmax set it for the figure, and they also drop the data outside that range.
FIGURE_ARGUMENTS = ("xmin", "xmax")


def get_directive_name(seriestype: str) -> str | None:
    """Return the name of the directive that a plot item gives, e.g. "ymin", or None for a series.

    A figure argument and an unknown directive stop the command.
    """
    # normalise_plotitems adds the underscore, thus report the name as the user wrote it
    given = seriestype.removeprefix("_")
    # a figure argument stops the command whichever way the caller spells it, because a plot item
    # of that name reaches the ion branch and gives an error that names no argument
    if given.lower() in FIGURE_ARGUMENTS:
        exit_with_error(
            f"'{given}' belongs to the whole figure and not to one subplot, because the subplots "
            f"share one horizontal axis. Give -{given.lower()} instead, which also drops the data outside"
        )

    if given.lower() == "xscale":
        exit_with_error(
            "'xscale' belongs to the whole figure and not to one subplot, because the subplots share one"
            " horizontal axis",
            "Give --logscalex for a log scale on the horizontal axis",
        )

    if seriestype.startswith("_") and given.lower() not in DIRECTIVES:
        suggestion = suggest_names(given, DIRECTIVES)
        exit_with_error(
            f"'{given}' is not a plot directive",
            f"{suggestion + ' ' if suggestion else ''}The directives are "
            f"{', '.join(f'{name}=' for name in DIRECTIVES)}",
        )

    return given.lower() if given.lower() in DIRECTIVES else None


def get_item_directive(item: str) -> str | None:
    """Return the name of the directive that an item of a subplot gives, e.g. "ymin" for "ymin=1e-16", or None."""
    name, equals, _ = item.partition("=")
    directive = name.removeprefix("_").lower()
    return directive if equals and directive in DIRECTIVES else None


def get_directive_value(subplot: Sequence[str], directive: str) -> str | None:
    """Return the value of a directive of a subplot, e.g. "log" for yscale=log, or None if the subplot has none."""
    return next((item.partition("=")[2] for item in reversed(subplot) if get_item_directive(item) == directive), None)


def replace_directives(subplot: Sequence[str], directives: Mapping[str, str | None]) -> tuple[str, ...]:
    """Return the items of a subplot with these directives in place of their old values. None removes a directive."""
    kept = [item for item in subplot if get_item_directive(item) not in directives]
    return (*kept, *(f"{name}={value}" for name, value in directives.items() if value is not None))


# A plot item that starts with this mark is a polars SQL expression of the estimator columns, as a formula of a
# spreadsheet, e.g. "=heating_coll - cooling_coll". No estimator variable, ion, or directive starts with "="
SQL_EXPRESSION_MARK: t.Final = "="

# a plot item that holds "pl." is the Python text of a polars expression, e.g. 'pl.col("Te") / 1e4'. No estimator
# variable, ion, or directive holds this text
PYTHON_EXPRESSION_PATTERN: t.Final = re.compile(r"\bpl\.")

# A series name can have its style after this mark, e.g. "Fe II@color=C3,linewidth=2". The mark needs no quotes
# in a shell, and zsh reads square brackets as a file pattern
SERIES_STYLE_MARK: t.Final = "@"


# the line styles that a series style accepts, by each name and short form, e.g. "--" gives "dashed"
SERIES_LINESTYLES: t.Final = MappingProxyType({name: name for name in LINESTYLE_NAMES} | LINESTYLE_ALIASES)


def get_series_style_problem(key: str, value: str) -> str | None:
    """Return the reason that a value of a series style is not valid, or None for a valid value.

    The colour, the dash pattern, and the width take the argparse types of the options of the other commands, e.g.
    color_arg of --color. An opacity of 0 is valid, because matplotlib then draws a clear line.
    """
    if key == "linestyle":
        return (
            None
            if value in SERIES_LINESTYLES
            else f"is not a line style. The line styles are {', '.join(SERIES_LINESTYLES)}"
        )
    if key == "linealpha":
        try:
            alpha = float(value)
        except ValueError:
            alpha = math.nan
        return None if 0.0 <= alpha <= 1.0 else "is not an opacity from 0 (clear) to 1 (opaque)"
    convert = {"color": color_arg, "dashes": dashes_arg, "linewidth": positive_float_arg}.get(key)
    try:
        if convert is not None:
            convert(value)
    except argparse.ArgumentTypeError as exc:
        return f"is not valid: {exc}"
    return None


def is_expression_text(text: str) -> bool:
    """Return True if the text is a polars expression in the SQL form, e.g. "=Te / 1e4", or in the Python form."""
    return text.startswith(SQL_EXPRESSION_MARK) or PYTHON_EXPRESSION_PATTERN.search(text) is not None


def partition_series_style(item: str) -> tuple[str, str, str]:
    """Return the name of a plot item, the style mark, and the text of the style, as str.partition gives them.

    The text of an expression can hold the mark, e.g. pl.col("Te").alias("T@e"). Thus the style of an expression starts
    at the first mark that a style key and an "=" follow. Each other item takes its first mark, thus an unknown key
    gives an error that names the style.
    """
    if not is_expression_text(item):
        return item.partition(SERIES_STYLE_MARK)
    # a mark inside a string of the expression is part of the string, e.g. pl.col("Te").alias("T@color=C3")
    openquote = ""
    escaped = False
    for index, char in enumerate(item):
        if openquote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == openquote:
                openquote = ""
        elif char in "\"'":
            openquote = char
        elif char == SERIES_STYLE_MARK and any(item.startswith(f"{key}=", index + 1) for key in SERIES_STYLE_KEYS):
            return item[:index], SERIES_STYLE_MARK, item[index + 1 :]
    return item, "", ""


def read_series_style(item: str) -> tuple[str, dict[str, str]]:
    """Return the name of a plot item and the style after its @, e.g. ("Fe II", {"color": "C3"}) for "Fe II@color=C3".

    The parts of the style have a comma between them. A part with no known key continues the value before it, thus a
    dash pattern can hold commas, e.g. "Te@dashes=5,2,linewidth=2". The label takes all the text after "label=", thus
    a label can hold a comma and an "=", e.g. "Te@color=C3,label=T, electrons". An item with no @ gives an empty style.
    A style that is not valid gives ValueError, and the message of the error names the problem.
    """
    name, mark, styletext = partition_series_style(item)
    style: dict[str, str] = {}
    if not mark:
        return item, style
    key: str | None = None
    parts = styletext.split(",")
    for index, part in enumerate(parts):
        partkey, equals, value = part.partition("=")
        if equals and partkey.strip() in SERIES_STYLE_KEYS:
            key = partkey.strip()
            if key == "label":
                style[key] = ",".join([value, *parts[index + 1 :]])
                break
            style[key] = value
        elif key is not None:
            style[key] += f",{part}"
        else:
            msg = (
                f"'{part}' in the plot item '{item}' is not a series style. Give key=value after the"
                f" {SERIES_STYLE_MARK}, e.g. '{name}{SERIES_STYLE_MARK}color=C3'. The keys are"
                f" {', '.join(SERIES_STYLE_KEYS)}"
            )
            raise ValueError(msg)
    for key, value in style.items():
        if key != "label" and (problem := get_series_style_problem(key, value)) is not None:
            msg = f"The {key} '{value}' of the plot item '{item}' {problem}"
            raise ValueError(msg)
    return name, style


def join_series_style(name: str, style: Mapping[str, str | None]) -> str:
    """Return the plot item of a series name with a style, e.g. "Fe II@color=C3". A value of None gives no key."""
    parts = [f"{key}={value}" for key in SERIES_STYLE_KEYS if (value := style.get(key)) is not None]
    return f"{name}{SERIES_STYLE_MARK}{','.join(parts)}" if parts else name


def get_series_name(item: str) -> str:
    """Return the name of a series item without its style, e.g. "Fe II" for "Fe II@color=C3"."""
    return partition_series_style(item)[0]


def get_item_style(item: str) -> dict[str, str]:
    """Return the style of a series item, e.g. {"color": "C3"} for "Fe II@color=C3".

    A style that is not valid gives no style here. plotestimators rejects it, and the status line gives the reason.
    """
    try:
        return read_series_style(item)[1]
    except ValueError:
        return {}


def set_item_style(subplot: Sequence[str], position: int, style: Mapping[str, str | None]) -> tuple[str, ...]:
    """Return the subplot with a new style for the series item at position. A value of None removes its key."""
    item = subplot[position]
    newitem = join_series_style(get_series_name(item), style)
    return (*subplot[:position], newitem, *subplot[position + 1 :])


def remove_item_styles(subplot: Sequence[str]) -> tuple[str, ...]:
    """Return the subplot with no style on its series items, e.g. "Fe II" for "Fe II@color=C3"."""
    return tuple(get_series_name(item) if get_item_directive(item) is None else item for item in subplot)


def resolve_item_alias(item: str) -> str:
    """Return an item with the variable in place of an alias, and with its style, e.g. nne@color=C3 for n_e@color=C3.

    plotestimators reads the alias as the variable, and the controls of the window follow the variable.
    """
    name = get_series_name(item)
    return VARIABLE_ALIASES.get(name, name) + item.removeprefix(name)


def split_plotitem_styles(plotitems: t.Any) -> tuple[list[t.Any], dict[str, dict[str, str]]]:
    """Return the plot items of a subplot without their series styles, and the style of each series name.

    A name that is an alias changes to its variable, e.g. n_e to nne, because the series of the plot use the variable.
    The items can be a plain list of words, or the nested form of main, e.g. [["populations", ["Fe II@color=C3"]]].
    """
    styles: dict[str, dict[str, str]] = {}

    def strip_style(item: t.Any) -> t.Any:
        if not isinstance(item, str):
            return item
        try:
            name, style = read_series_style(item)
        except ValueError as exc:
            exit_with_error(str(exc))
        if style:
            styles[VARIABLE_ALIASES.get(name, name)] = style
        return name

    if isinstance(plotitems, str):
        plotitems = [plotitems]
    stripped = [
        [item[0], [strip_style(name) for name in item[1]]]
        if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], (list, tuple))
        else strip_style(item)
        for item in plotitems
    ]
    return stripped, styles


def get_styles_of_subplots(args: argparse.Namespace, count: int) -> list[dict[str, dict[str, str]]]:
    """Return the style of each series name for each of count subplots, see split_plotitem_styles.

    The default plot list gives no style, thus a subplot with no plot item of the user has an empty style.
    """
    styles = [split_plotitem_styles(plotitems)[1] for plotitems in (args.plotlist or [])[:count]]
    return [*styles, *({} for _ in range(count - len(styles)))]


def is_expression_item(item: object) -> bool:
    """Return True if a plot item is a polars expression, e.g. "=Te / 1e4@color=C3" or 'pl.col("Te") / 1e4'."""
    return isinstance(item, str) and is_expression_text(get_series_name(item))


def starts_like_a_flag(item: str) -> bool:
    """Return True if argparse can read the start of a plot item as a flag, e.g. -pl.col("TR") as -p.

    A letter after the minus sign gives a flag. An item with a space and no letter there gives a plot item, e.g.
    - pl.col("TR") or -1 * pl.col("TR"). Python 3.13 reads an item with no space as a flag, e.g. -.5*pl.col("TR").
    """
    return re.match(r"-(?:[A-Za-z]|\S*$)", item) is not None


def evaluate_expression_text(text: str) -> pl.Expr:
    """Return the polars expression of the text of a plot item.

    The SQL form is polars SQL after the "=", e.g. "=heating_coll - cooling_coll", and it runs no Python code. Text
    that gives no expression gives ValueError, and a Python result that is not an expression gives TypeError.
    """
    try:
        if text.startswith(SQL_EXPRESSION_MARK):
            expr = pl.sql_expr(text.removeprefix(SQL_EXPRESSION_MARK))
        else:
            # the user typed the text, thus it runs with the rights of the user. The name pl alone is in scope
            expr = eval(text, {"__builtins__": {}, "pl": pl})  # ruff:ignore[suspicious-eval-usage]
    # the text can raise any exception, e.g. IndexError for pl.col(["Te"][1])
    except Exception as exc:
        # a polars SQL error holds the text again on more lines, with a mark under the problem
        reason = next((line for line in str(exc).splitlines() if line.strip()), type(exc).__name__)
        msg = f"The plot item '{text}' is not a valid polars expression: {reason}"
        raise ValueError(msg) from exc
    if not isinstance(expr, pl.Expr):
        msg = f"The plot item '{text}' gives a {type(expr).__name__} and not a polars expression"
        raise TypeError(msg)
    return expr


EXPRESSION_HINT: t.Final = (
    "Give polars SQL after an =, e.g. =heating_coll - cooling_coll, or Python text with pl as the polars module, e.g."
    ' pl.col("heating_coll") - pl.col("cooling_coll")'
)


def get_expression_problem(text: str, estimatorcolumns: Collection[str]) -> tuple[str, str] | None:
    """Return the problem of the text of an expression item and a hint, or None for an expression that gives a series.

    The text holds no style. The command stops with the problem.
    """
    try:
        expr = evaluate_expression_text(text)
    except (ValueError, TypeError) as exc:
        return str(exc), EXPRESSION_HINT
    return get_evaluated_expression_problem(expr, text, estimatorcolumns)


def get_evaluated_expression_problem(
    expr: pl.Expr, text: str, estimatorcolumns: Collection[str]
) -> tuple[str, str] | None:
    """Return the problem of an expression that the text of a plot item gave and a hint, or None for a valid series."""
    if absentcolumns := [name for name in expr.meta.root_names() if name not in estimatorcolumns]:
        # a style with an unknown key stays in the text of an expression, see partition_series_style
        stylehint = (
            f"The keys of a style after {SERIES_STYLE_MARK} are {', '.join(SERIES_STYLE_KEYS)}"
            if SERIES_STYLE_MARK in absentcolumns[0]
            else ""
        )
        return (
            f"The plot item '{text}' reads '{absentcolumns[0]}', which is not an estimator variable",
            stylehint
            or suggest_names(absentcolumns[0], estimatorcolumns)
            or "Run with --listvariables to see the variables of this model",
        )
    try:
        results = [run_on_sample_frame(expr, estimatorcolumns, rowcount) for rowcount in SAMPLE_ROWCOUNTS]
        keepsroworder = keeps_row_order(expr, estimatorcolumns)
    except pl.exceptions.PolarsError as exc:
        # with the names alone, each column takes the type Float64, thus an error can come from that guess
        if not isinstance(estimatorcolumns, pl.Schema):
            return None
        reason = next((line for line in str(exc).splitlines() if line.strip()), type(exc).__name__)
        return (
            f"The plot item '{text}' does not suit the types of the estimator columns: {reason}",
            "Give an expression of the columns that gives a number, e.g. =Te / 1e4",
        )
    result = results[0]
    if result.width != 1:
        hint = (
            "Give one plot item for each column, e.g. =Te and =TR, or join the columns, e.g."
            ' pl.sum_horizontal(pl.col("^heating_.*$"))'
        )
        return f"The plot item '{text}' gives {result.width} columns, and a series takes one column", hint
    # a series has one value for each row of the estimators, or one value for all the rows
    if any(
        sampleresult.height not in {1, rowcount}
        for sampleresult, rowcount in zip(results, SAMPLE_ROWCOUNTS, strict=True)
    ):
        problem = (
            f"The plot item '{text}' changes the number of rows, e.g. with a filter, and a series takes one value for"
            " each cell and timestep"
        )
        return problem, 'Give a condition with pl.when, e.g. pl.when(pl.col("Te") > 1e4).then(pl.col("Te"))'
    # the plot puts the values next to the cells by position, thus a sort gives each cell the value of a different cell
    if not keepsroworder:
        problem = (
            f"The plot item '{text}' changes the order of the rows, e.g. with sort(), reverse(), or shift(), and each"
            " value of a series belongs to its own cell and timestep"
        )
        return problem, 'Give an expression of each row, or an aggregation, e.g. pl.col("Te") - pl.col("Te").mean()'
    # the plot multiplies the values by the weights, thus a text gives no series. A boolean gives 0 and 1
    if not ((dtype := result.dtypes[0]).is_numeric() or dtype == pl.Boolean):
        return (
            f"The plot item '{text}' gives values of the type {dtype}, and a series takes numbers",
            "Give an expression of the columns that gives a number, e.g. =Te / 1e4",
        )
    return None


# the numbers of rows of the sample frames. A head(n) or a slice gives a fixed number of rows, thus a series must keep
# two different numbers of rows
SAMPLE_ROWCOUNTS: t.Final = (23, 4099)


def get_sample_values(rowcount: int, columnposition: int) -> list[float | None]:
    """Return the values of a column of a sample frame: a null, zeros, negative values, and repeated values.

    A filter or a unique() then changes the number of rows, which a series must keep. Each column has a different shift
    of the values, thus a comparison of two columns, e.g. pl.col("Te") >= pl.col("TR"), changes from row to row.
    """
    return [*(float((index + 3 * columnposition) % 7 - 3) for index in range(rowcount - 1)), None]


def get_column_schema(estimatorcolumns: Collection[str]) -> pl.Schema:
    """Return the schema of the estimator columns. A list of names gives each column the type Float64."""
    if isinstance(estimatorcolumns, pl.Schema):
        return estimatorcolumns
    return pl.Schema(dict.fromkeys(estimatorcolumns, pl.Float64))


def get_input_columns(expr: pl.Expr, estimatorcolumns: Collection[str]) -> list[str]:
    """Return the estimator columns that an expression reads. A fault, e.g. an absent column, gives a PolarsError.

    polars names no root column for a selector, e.g. pl.col("^heating_.*$"). Thus each selection in the expression runs
    on the columns of the model.
    """
    schema = get_column_schema(estimatorcolumns)
    inputcolumns: dict[str, None] = {}
    pending = [expr]
    while pending:
        node = pending.pop()
        if node.meta.is_column_selection(allow_aliasing=True):
            # an alias names the output, e.g. temperature in pl.col("Te").alias("temperature"), and not the column
            selection = node.meta.undo_aliases()
            inputcolumns.update(dict.fromkeys(pl.LazyFrame(schema=schema).select(selection).collect_schema().names()))
        else:
            pending.extend(node.meta.pop())
    return list(inputcolumns)


def get_sample_column(name: str, dtype: pl.DataType, rowcount: int, columnposition: int) -> pl.Series:
    """Return a column of a sample frame with the type of an estimator column, or a column of nulls of that type."""
    try:
        return pl.Series(name, get_sample_values(rowcount, columnposition), dtype=pl.Float64).cast(dtype, strict=False)
    except pl.exceptions.PolarsError:
        return pl.Series(name, [None] * rowcount, dtype=dtype)


def run_on_sample_frame(
    expr: pl.Expr, estimatorcolumns: Collection[str], rowcount: int = SAMPLE_ROWCOUNTS[0]
) -> pl.DataFrame:
    """Return the result of an expression on a sample frame of the columns that it reads. A fault gives a PolarsError.

    The schema alone does not find a type error, e.g. a str method on a float, or a filter that changes the number of
    rows. A pl.Schema gives the types of the columns, and each column of a list of names takes the type Float64.
    """
    return get_sample_frame(expr, estimatorcolumns, rowcount).lazy().select(expr).collect()


def get_sample_frame(expr: pl.Expr, estimatorcolumns: Collection[str], rowcount: int) -> pl.DataFrame:
    """Return a sample frame of the columns that an expression reads, with the types of the estimator columns."""
    schema = get_column_schema(estimatorcolumns)
    inputcolumns = get_input_columns(expr, schema)
    samplecolumns = [
        get_sample_column(name, schema[name], rowcount, columnposition)
        for columnposition, name in enumerate(inputcolumns)
    ]
    # an expression that reads no column can still give one value for each row, e.g. pl.repeat(1.0, pl.len()). A
    # column with an empty name gives the rows, because a selector of the expression, e.g. "^cooling_.*$", cannot
    # match it
    return pl.DataFrame(samplecolumns or [pl.Series("", range(rowcount))])


# a fixed order of the rows of a sample frame with no pattern. A rotation of the rows commutes with a cyclic gather
SAMPLE_PERMUTATION: t.Final = (8, 2, 19, 13, 0, 21, 5, 16, 11, 3, 22, 7, 14, 1, 18, 9, 20, 6, 12, 4, 17, 10, 15)


def keeps_row_order(expr: pl.Expr, estimatorcolumns: Collection[str]) -> bool:
    """Return True if each value of an expression belongs to its own row, or if the expression gives one value.

    An expression of each row, or an aggregation, gives the same values for a sample with its rows in a different
    order, in that order. A sort, a reverse(), a shift(), a gather, or a cumulative sum gives different values.
    """
    sample = get_sample_frame(expr, estimatorcolumns, len(SAMPLE_PERMUTATION))
    permutation = list(SAMPLE_PERMUTATION)
    reorderedsample = sample[permutation]
    result = sample.lazy().select(expr).collect()
    # get_evaluated_expression_problem reports a result of more or less than one column, or of a different length
    if result.width != 1 or result.height != sample.height:
        return True
    values = result.to_series()
    reorderedvalues = reorderedsample.lazy().select(expr).collect().to_series()
    expected = values.gather(permutation)
    if not (values.dtype.is_numeric() or values.dtype == pl.Boolean):
        return reorderedvalues.equals(expected)
    # the sample values are of order 1, thus a sum in a different order differs by approximately 1e-16
    reordered, wanted = reorderedvalues.cast(pl.Float64).to_numpy(), expected.cast(pl.Float64).to_numpy()
    return np.allclose(reordered, wanted, rtol=0, atol=1e-9, equal_nan=True)


def get_typed_item_problem(item: str, estimatorcolumns: Collection[str]) -> tuple[str, str] | None:
    """Return the problem of an expression item that the user types in the window, and the problem with a hint.

    The window shows the problem before it adds the item. Each item that is not an expression gives None, because the
    command finds its problems. polars SQL runs no code, thus the window checks it fully, and Python text gets a check
    of its syntax alone.
    """
    if not is_expression_item(item):
        return None
    if starts_like_a_flag(item):
        problem = f"The command reads {item[:2]} as a flag"
        return problem, f"{problem}. Write 0 {item.replace('-', '- ', 1)} in place of {item}."
    try:
        name, _ = read_series_style(item)
    except ValueError as exc:
        return str(exc), str(exc)
    if not name.startswith(SQL_EXPRESSION_MARK):
        # the window runs no Python text. The command runs it, on the model host for a remote model, and gives the
        # other problems
        try:
            compile(name, "<plot item>", "eval")
        except SyntaxError as exc:
            problem = f"The plot item '{name}' is not valid Python: {exc.msg}"
            return problem, f"{problem}. {EXPRESSION_HINT}"
        return None
    if (expressionproblem := get_expression_problem(name, estimatorcolumns)) is None:
        return None
    problem, hint = expressionproblem
    return problem, f"{problem}. {hint}"


def parse_expression_item(text: str, estimatorcolumns: Collection[str]) -> pl.Expr:
    """Return the polars expression of the text of a plot item, with the text as the name of the expression.

    The text names the series in the styles of the subplot, see split_plotitem_styles. get_series_variables gives the
    units and the weight of the series.
    """
    try:
        expr = evaluate_expression_text(text)
    except (ValueError, TypeError) as exc:
        exit_with_error(str(exc), EXPRESSION_HINT)
    if (problem := get_evaluated_expression_problem(expr, text, estimatorcolumns)) is not None:
        exit_with_error(*problem)
    return expr.alias(text)


# a name that SQL reads without quotes, e.g. heating_coll. A name such as heating_dep/total_dep needs double quotes
SQL_PLAIN_NAME_PATTERN: t.Final = re.compile(r"[A-Za-z_]\w*")


def get_expression_name_start(text: str) -> tuple[str, str] | None:
    """Return the text before the column name at the end of an expression, and the start of that name, or None.

    The window completes the name. In the Python form, the name follows pl.col(" or pl.col('. In the SQL form, the
    name is the last word, or the text after a double quote with no end.
    """
    if not text.startswith(SQL_EXPRESSION_MARK):
        match = re.fullmatch(r"(?s)(.*\bpl\.col\(\s*[\"'])([^\"']*)", text)
        return None if match is None else (match[1], match[2])
    if text.count('"') % 2 == 1:
        head, _, start = text.rpartition('"')
        return f'{head}"', start
    match = re.fullmatch(r"(?s)(.*?)([A-Za-z_]\w*)?", text)
    if match is None:
        return None
    head, start = match[1], match[2] or ""
    # a name cannot follow a number, a closing bracket, or a closed quote, e.g. in =Te * 1e4 or =LOG10(Te)
    before = head[-1:] if start else head.rstrip()[-1:]
    return None if before.isdigit() or before in {".", ")", '"'} else (head, start)


def complete_expression_name(text: str, name: str) -> str:
    """Return the text of an expression with the column name in place of the start of the name at its end.

    The Python form closes the call, e.g. pl.col("heating_coll"). The SQL form puts a name in double quotes when SQL
    needs them, e.g. "heating_dep/total_dep".
    """
    if (parts := get_expression_name_start(text)) is None:
        return text
    head = parts[0]
    if not text.startswith(SQL_EXPRESSION_MARK):
        return f"{head}{name}{head[-1]})"
    if head.endswith('"') and head.count('"') % 2 == 1:
        return f'{head}{name}"'
    return f"{head}{name}" if SQL_PLAIN_NAME_PATTERN.fullmatch(name) else f'{head}"{name}"'


def get_series_variables(colexpr: pl.Expr, estimatorcolumns: Collection[str]) -> tuple[str, str, str]:
    """Return the name of the series of an expression, the variable of its units, and the variable of its weight.

    The text of an expression item names its series, e.g. 'pl.col("Te") / 1e4'. Only a plain column keeps the units
    of its variable, because an operation can change the units. The first column gives the weight of the mean, as for
    an expression with no alias. Thus 'pl.col("init_X_Fe56")' takes the cell mass as the weight.
    """
    seriesname = colexpr.meta.output_name()
    if not is_expression_item(seriesname):
        return seriesname, seriesname, seriesname
    unaliased = colexpr.meta.undo_aliases()
    inputcolumns: list[str] = []
    outputnames: list[str] = []
    # get_expression_problem gives the error of an expression that polars cannot resolve. Without the columns, the
    # series takes the text as the variable of its units and of its weight
    with contextlib.suppress(pl.exceptions.PolarsError):
        inputcolumns = get_input_columns(unaliased, estimatorcolumns)
        outputnames = run_on_sample_frame(unaliased, estimatorcolumns).columns
    # polars names 2 * pl.col("init_X_Fe56") after the literal, and root_names() leaves out a selector, e.g. in
    # pl.sum_horizontal(pl.col("^init_X_.*$")) + pl.col("Te"). get_input_columns gives the columns from the left
    firstcolumn = next(iter(inputcolumns), None) or next(iter(unaliased.meta.root_names()), seriesname)
    # a selection of one column, e.g. pl.col("Te") or pl.col("^Te$"), is the variable itself
    isonecolumn = unaliased.meta.is_column_selection() and len(outputnames) == 1
    return seriesname, firstcolumn if isonecolumn else seriesname, firstcolumn


def get_series_items(subplot: Sequence[str]) -> list[str]:
    """Return the items of a subplot that are not a directive, with the style of each series, e.g. "Fe II@color=C3"."""
    return [item for item in subplot if get_item_directive(item) is None]


def get_subplot_names(subplot: Sequence[str]) -> list[str]:
    """Return the names of a subplot without the directives and the styles.

    The names are e.g. the variables, or a series type and its ions.
    """
    return [get_series_name(item) for item in get_series_items(subplot)]


def get_seriestype_of_names(names: Sequence[t.Any], estimatorcolumns: Collection[str]) -> str | None:
    """Return the type of series of the names of a subplot, e.g. "populations", or None for a subplot of variables.

    The caller removes each directive and each style from the names, and it changes each alias to its variable. A type
    of series comes first in the names, e.g. "populations Fe II". A list of ions with no type is a plot of populations.
    The function tests for this case last, thus an estimator column first in the list gives a subplot of variables,
    e.g. "Te" and not tellurium.
    """
    if not names:
        return None
    first = names[0]
    if not isinstance(first, str) or is_expression_item(first):
        return None
    if is_seriestype(first, estimatorcolumns) or is_ionseriestype(first, estimatorcolumns, names[1:]):
        return first
    if first not in estimatorcolumns and all(map(could_be_ion, names)):
        return "populations"
    return None


def get_subplot_seriestype(subplot: Sequence[str], estimatorcolumns: Collection[str]) -> str | None:
    """Return the type of series of a subplot, e.g. "populations", or None for a subplot of variables.

    normalise_plotitems uses the same rule. Thus the viewer gives a subplot the type that the command reads, also
    for a subplot that the command then rejects, e.g. ("Fe II", "te").
    """
    names = [VARIABLE_ALIASES.get(name, name) for name in get_subplot_names(subplot)]
    return get_seriestype_of_names(names, estimatorcolumns)


def normalise_plotitems(plotitems: t.Any, estimatorcolumns: Collection[str]) -> list[t.Any]:
    """Resolve variable aliases and move any 'key=value' plot directives to the end of the plot item list.

    A list of ions such as ["Sr I", "Sr II"] is rewritten as a populations plot [["populations", ["Sr I", "Sr II"]]].
    The text of a polars expression becomes the expression, see parse_expression_item.
    """
    # the client draws the series styles, see draw_line_figure, thus the data needs the names alone
    plotitems, _ = split_plotitem_styles(plotitems)

    # an expression can hold an "=", e.g. pl.col("Te").clip(lower_bound=0), and it is never a directive
    isdirective = [
        isinstance(plotvar, str) and "=" in plotvar and not is_expression_item(plotvar) for plotvar in plotitems
    ]
    # the underscore of a directive is optional, thus "-p rho yscale=log" and "_yscale=log" are the same.
    # One shape here lets get_subplot_plans find an unknown directive whichever spelling the user gave
    plot_directives = [
        ["_" + key.removeprefix("_"), value]
        for key, value in (
            plotvar.split("=", maxsplit=1)
            for plotvar, directive in zip(plotitems, isdirective, strict=True)
            if directive
        )
    ]
    plotvars = [
        parse_expression_item(plotvar, estimatorcolumns)
        if is_expression_item(plotvar)
        else VARIABLE_ALIASES.get(plotvar, plotvar)
        if isinstance(plotvar, str)
        else plotvar
        for plotvar, directive in zip(plotitems, isdirective, strict=True)
        if not directive
    ]

    # a subplot with no series draws an empty frame, e.g. in the viewer after the user removed its last series
    if not plotvars:
        return plot_directives

    seriestype = get_seriestype_of_names(plotvars, estimatorcolumns)
    if seriestype is not None and any(isinstance(plotvar, pl.Expr) for plotvar in plotvars):
        exit_with_error(
            f"A polars expression gives a series of a subplot of variables, and this subplot holds {seriestype}",
            "Give the expression a --plot of its own",
        )
    # the grouped form names the type of series first, e.g. -plot populations "Fe II" "Fe III"
    if seriestype is not None and seriestype == plotvars[0]:
        if len(plotvars) == 1:
            exit_with_error(
                f"'{plotvars[0]}' names a type of series and takes at least one name after it",
                f'e.g. --plot {plotvars[0]} Fe. Quote an ion that holds a space, e.g. --plot {plotvars[0]} "Fe II"',
            )
        if all(isinstance(plotvar, str) for plotvar in plotvars[1:]):
            plotvars = [[plotvars[0], plotvars[1:]]]
    elif seriestype == "populations":
        # an ion population plot is the reading of last resort, thus reject a name that is no ion at all
        if notions := [var for var in plotvars if isinstance(var, str) and not is_valid_ion(var)]:
            exit_with_error(
                f"'{notions[0]}' is neither an estimator variable nor an ion",
                suggest_names(notions[0], estimatorcolumns) or "Run with --listvariables to see the variables",
            )

        # Te and W name an estimator variable and also an element. A model with no estimator files has no such column,
        # and the plot then read tellurium in place of the temperature and drew an empty subplot
        if absentvariables := [
            var for var in plotvars if isinstance(var, str) and var in VARIABLES and var not in estimatorcolumns
        ]:
            exit_with_error(
                f"'{absentvariables[0]}' is an estimator variable, and the estimators of this model hold no such"
                " column",
                "Run with --listvariables to see the variables of this model",
            )

        # plotting this as a variable would cause an error, so interpret it as ion populations instead
        new_plotvars = [["populations", plotvars]]
        print(f"Rewriting plotlist {plotvars} to {new_plotvars}")
        plotvars = new_plotvars

    return plotvars + plot_directives


def get_plotitem_tokens(plotitems: Sequence[t.Any]) -> tuple[str, ...]:
    """Return the --plot tokens of the items of one subplot of the default plot list.

    A directive has the form ["_yscale", "log"], and it becomes "yscale=log". A type of series has the form
    ["populations", ["Fe II", "Fe III"]], and its names follow it. A polars expression gives its output name, which
    --plot cannot read.
    """
    tokens: list[str] = []
    for plotitem in plotitems:
        if isinstance(plotitem, str):
            tokens.append(plotitem)
        elif isinstance(plotitem, pl.Expr):
            tokens.append(plotitem.meta.output_name())
        elif str(plotitem[0]).startswith("_"):
            tokens.append(f"{str(plotitem[0]).removeprefix('_')}={plotitem[1]}")
        else:
            tokens.extend([str(plotitem[0]), *(str(name) for name in plotitem[1])])
    return tuple(tokens)


def get_python_number(text: str) -> float | str:
    """Return the number that a directive value gives, e.g. 1e-16 for "1e-16", or the text if it gives none."""
    with contextlib.suppress(ValueError):
        return int(text)
    with contextlib.suppress(ValueError):
        if math.isfinite(number := float(text)):
            return number
    return text


def get_python_plotitems(subplot: Sequence[str], estimatorcolumns: Collection[str]) -> list[t.Any]:
    """Return a subplot in the form of an item of the plotlist of main.

    A type of series groups its names, e.g. [["populations", ["Fe II", "Fe III"]]], and a directive follows the
    names, e.g. ["_ymin", 1e-16].
    """
    # main reads the style of a series in its name, e.g. "Fe II@color=C3", as the command does
    items = get_series_items(subplot)
    seriestype = get_subplot_seriestype(subplot, estimatorcolumns)
    # a list of ions with no type is a populations subplot, and its names hold no type
    series: list[t.Any] = (
        items if seriestype is None else [[seriestype, items[1:] if items[0] == seriestype else items]]
    )
    directives = [
        [
            f"_{directive}",
            get_python_number(item.partition("=")[2]) if directive in {"ymin", "ymax"} else item.partition("=")[2],
        ]
        for item in subplot
        if (directive := get_item_directive(item)) is not None
    ]
    return [*series, *directives]
