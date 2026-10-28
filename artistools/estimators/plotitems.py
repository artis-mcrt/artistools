"""Read and write the plot items of plotestimators, e.g. the variables, the ions, the directives, and the styles.

The command and its viewer both use this module, thus the viewer reads each subplot as the command does. A subplot of
the viewer holds the --plot tokens of the command, e.g. ("populations", "Fe II@color=C3", "yscale=log").
"""

import argparse
import contextlib
import math
import typing as t
from collections.abc import Collection
from collections.abc import Mapping
from collections.abc import Sequence
from types import MappingProxyType

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
            return get_element_or_ion_tuple(ionstr)

    # a name that is no ion at all, e.g. a mistyped variable. The caller tests the atomic number
    return (get_atomic_number(ionstr.split("_", maxsplit=1)[0]), ionstr)


def get_element_or_ion_tuple(ionstr: str) -> tuple[int, str | int]:
    """Return the atomic number and the ion stage of a name that get_ion_tuple reads, or "ALL" for an element."""
    iontuple = get_ion_tuple(ionstr)
    return (iontuple, "ALL") if isinstance(iontuple, int) else iontuple


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

    if not isinstance(plotvar, str):
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
        alpha = float(value) if is_float(value) else math.nan
        return None if 0.0 <= alpha <= 1.0 else "is not an opacity from 0 (clear) to 1 (opaque)"
    convert = {"color": color_arg, "dashes": dashes_arg, "linewidth": positive_float_arg}.get(key)
    try:
        if convert is not None:
            convert(value)
    except argparse.ArgumentTypeError as exc:
        return f"is not valid: {exc}"
    return None


def is_float(text: str) -> bool:
    """Return True if float() reads the text."""
    try:
        float(text)
    except ValueError:
        return False
    return True


def read_series_style(item: str) -> tuple[str, dict[str, str]]:
    """Return the name of a plot item and the style after its @, e.g. ("Fe II", {"color": "C3"}) for "Fe II@color=C3".

    The parts of the style have a comma between them. A part with no known key continues the value before it, thus a
    dash pattern can hold commas, e.g. "Te@dashes=5,2,linewidth=2". The label takes all the text after "label=", thus
    a label can hold a comma and an "=", e.g. "Te@color=C3,label=T, electrons". An item with no @ gives an empty style.
    A style that is not valid gives ValueError, and the message of the error names the problem.
    """
    name, mark, styletext = item.partition(SERIES_STYLE_MARK)
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


def split_series_style(item: str) -> tuple[str, dict[str, str]]:
    """Return the name and the style of a plot item, or stop the command if the style is not valid."""
    try:
        return read_series_style(item)
    except ValueError as exc:
        exit_with_error(str(exc))


def join_series_style(name: str, style: Mapping[str, str | None]) -> str:
    """Return the plot item of a series name with a style, e.g. "Fe II@color=C3". A value of None gives no key."""
    parts = [f"{key}={value}" for key in SERIES_STYLE_KEYS if (value := style.get(key)) is not None]
    return f"{name}{SERIES_STYLE_MARK}{','.join(parts)}" if parts else name


def get_series_name(item: str) -> str:
    """Return the name of a series item without its style, e.g. "Fe II" for "Fe II@color=C3"."""
    return item.partition(SERIES_STYLE_MARK)[0]


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
        name, style = split_series_style(item)
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

    The names have no directive and no style, and an alias has changed to its variable. A type of series comes first
    in the names, e.g. "populations Fe II". A list of ions with no type is a plot of populations. This reading comes
    last, thus an estimator column first in the list gives a subplot of variables, e.g. "Te" and not tellurium.
    """
    if not names:
        return None
    first = names[0]
    if not isinstance(first, str):
        return None
    if is_seriestype(first, estimatorcolumns) or is_ionseriestype(first, estimatorcolumns, names[1:]):
        return first
    if first not in estimatorcolumns and all(map(could_be_ion, names)):
        return "populations"
    return None


def get_subplot_seriestype(subplot: Sequence[str], estimatorcolumns: Collection[str]) -> str | None:
    """Return the type of series of a subplot, e.g. "populations", or None for a subplot of variables.

    normalise_plotitems uses the same reading. Thus the viewer gives a subplot the type that the command reads, also
    for a subplot that the command then rejects, e.g. ("Fe II", "te").
    """
    names = [VARIABLE_ALIASES.get(name, name) for name in get_subplot_names(subplot)]
    return get_seriestype_of_names(names, estimatorcolumns)


def normalise_plotitems(plotitems: t.Any, estimatorcolumns: Collection[str]) -> list[t.Any]:
    """Resolve variable aliases and move any 'key=value' plot directives to the end of the plot item list.

    A list of ions such as ["Sr I", "Sr II"] is rewritten as a populations plot [["populations", ["Sr I", "Sr II"]]].
    """
    # the client draws the series styles, see draw_line_figure, thus the data needs the names alone
    plotitems, _ = split_plotitem_styles(plotitems)

    # the underscore of a directive is optional, thus "-p rho yscale=log" and "_yscale=log" are the same.
    # One shape here lets get_subplot_plans find an unknown directive whichever spelling the user gave
    plot_directives = [
        ["_" + key.removeprefix("_"), value]
        for key, value in (
            plotvar.split("=", maxsplit=1) for plotvar in plotitems if isinstance(plotvar, str) and "=" in plotvar
        )
    ]
    plotvars = [
        VARIABLE_ALIASES.get(plotvar, plotvar) if isinstance(plotvar, str) else plotvar
        for plotvar in plotitems
        if not isinstance(plotvar, str) or "=" not in plotvar
    ]

    # a subplot with no series draws an empty frame, e.g. in the viewer after the user removed its last series
    if not plotvars:
        return plot_directives

    seriestype = get_seriestype_of_names(plotvars, estimatorcolumns)
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
    ["populations", ["Fe II", "Fe III"]], and its names follow it.
    """
    tokens: list[str] = []
    for plotitem in plotitems:
        if isinstance(plotitem, str):
            tokens.append(plotitem)
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
