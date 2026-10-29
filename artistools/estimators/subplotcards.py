"""Show the subplots of the plotestimators window as cards, and change the subplots of the window.

A subplot of the window holds the --plot tokens of one subplot of the command, e.g. ("Te", "TR", "yscale=log"). The
pure functions change these tuples, and add_subplot_section makes the cards and their handlers.
"""

import argparse
import dataclasses as dc
import shlex
import typing as t
from functools import lru_cache
from functools import partial
from pathlib import Path
from types import MappingProxyType

import matplotlib as mpl
import numpy as np
import polars as pl
from matplotlib.collections import QuadMesh

from artistools.atomic import get_ionstring
from artistools.estimators.core import format_units
from artistools.estimators.core import get_prefix_group
from artistools.estimators.core import PREFIX_GROUPS
from artistools.estimators.core import split_species_suffix
from artistools.estimators.plotestimators import get_panel_axes_label
from artistools.estimators.plotestimators import get_ylabel
from artistools.estimators.plotestimators import POPTYPE_YLABELS
from artistools.estimators.plotitems import get_directive_value
from artistools.estimators.plotitems import get_iontuple
from artistools.estimators.plotitems import get_iontuple_sortkey
from artistools.estimators.plotitems import get_item_directive
from artistools.estimators.plotitems import get_item_style
from artistools.estimators.plotitems import get_series_items
from artistools.estimators.plotitems import get_series_name
from artistools.estimators.plotitems import get_subplot_names
from artistools.estimators.plotitems import get_subplot_seriestype
from artistools.estimators.plotitems import is_seriestype
from artistools.estimators.plotitems import is_valid_ion
from artistools.estimators.plotitems import remove_item_styles
from artistools.estimators.plotitems import replace_directives
from artistools.estimators.plotitems import resolve_item_alias
from artistools.estimators.plotitems import set_item_style
from artistools.nltepops import read_nltepops
from artistools.plottools import plain_label
from artistools.viewertools.core import get_row_values
from artistools.viewertools.core import OptionRows
from artistools.viewertools.core import run_command_step
from artistools.viewertools.sections import read_limit_fields
from artistools.viewertools.series import edit_series_properties
from artistools.viewertools.series import make_line_swatch
from artistools.viewertools.series import make_swatch_button
from artistools.viewertools.widgets import add_section
from artistools.viewertools.widgets import get_accent_colour
from artistools.viewertools.widgets import make_completer
from artistools.viewertools.widgets import make_drag_header
from artistools.viewertools.widgets import make_flow_layout
from artistools.viewertools.widgets import make_glyph_button
from artistools.viewertools.widgets import make_grip
from artistools.viewertools.widgets import make_note_label
from artistools.viewertools.widgets import make_row_layout
from artistools.viewertools.widgets import set_disclosure_symbol
from artistools.viewertools.widgets import set_edit_text
from artistools.viewertools.widgets import set_note_text

if t.TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Mapping
    from collections.abc import Sequence

    from PySide6 import QtGui
    from PySide6 import QtWidgets

    from artistools.estimators.interactive import ControlValues
    from artistools.estimators.interactive import EstimatorViewer
    from artistools.viewertools.window import DrawQueue


# the number of levels of each ion that the choices of a level population give, from the lowest NLTE level up
LEVEL_CHOICES_PER_ION: t.Final = 20


def get_level_names(modelpath: Path, timestep: int, cell: int) -> list[str]:
    """Return the names of the lowest NLTE levels of each ion for a plot of level populations, e.g. "Fe II 0".

    Each cell has the same NLTE levels, thus one cell gives them. A timestep before the first NLTE timestep has no
    populations, and all the timesteps of the cell then give the levels.
    """
    dfpops = read_nltepops(modelpath, timestep=timestep, modelgridindex=cell)
    if dfpops.is_empty():
        dfpops = read_nltepops(modelpath, modelgridindex=cell)
    # a negative level is the superlevel of an ion, which is not one level
    dflevels = (
        dfpops
        .filter(pl.col("level") >= 0)
        .select("Z", "ion_stage", "level")
        .unique()
        .sort("Z", "ion_stage", "level")
        .group_by("Z", "ion_stage", maintain_order=True)
        .head(LEVEL_CHOICES_PER_ION)
    )
    return [
        f"{get_ionstring(atomic_number, ion_stage)} {level}"
        for atomic_number, ion_stage, level in zip(dflevels["Z"], dflevels["ion_stage"], dflevels["level"], strict=True)
    ]


# the variables that suit a new subplot, in the order of the suggestions
COMMON_VARIABLES: t.Final = ("Te", "TR", "nne", "rho", "heating_dep", "total_dep", "TJ", "W")


def get_first_common_variable(estimatorcolumns: "Collection[str]") -> str:
    """Return the first variable of COMMON_VARIABLES that the model holds, or else Te."""
    return next((name for name in COMMON_VARIABLES if name in estimatorcolumns), "Te")


# the columns that give the grid, the time, or the size of a cell, and not the physics of the cell
BOOKKEEPING_COLUMNS: t.Final = frozenset({
    "deltavol_deltat",
    "inputcellid",
    "mass_g",
    "modelgridindex",
    "tdays",
    "thick",
    "timestep",
    "titeration",
    "tmid_days",
    "tmid_days_prevtimestep",
    "tstart_days",
    "twidth_days",
    "volume_prevtimestep",
})


# the families of the estimator columns that give the species of each type of series
SPECIES_FAMILIES: t.Final = MappingProxyType({
    "populations": ("nnelement", "nnion", "nniso"),
    "averageionisation": ("nnelement",),
    "averageexcitation": ("nnion",),
    "initabundances": ("init_X",),
    "initmasses": ("init_X",),
})


# the types of series that read the NLTE populations of the model, and not the estimators alone
NLTE_SERIESTYPES: t.Final = frozenset({"averageexcitation", "levelpopulation", "levelpopulation_dn_on_dvel"})


# the number of suggestions of the card of a subplot, and of the new subplots
SERIES_SUGGESTION_COUNT: t.Final = 4


NEW_SUBPLOT_SUGGESTION_COUNT: t.Final = 5


def get_species_sortkey(species: str) -> tuple[int, int, int, str]:
    """Return the key that sorts species by element, then the element before its ions, then the ions by stage."""
    return get_iontuple_sortkey(get_iontuple(species))


@lru_cache(maxsize=4)
def get_species_of_families(estimatorcolumns: tuple[str, ...]) -> "Mapping[str, tuple[str, ...]]":
    """Return the species of each family of estimator columns in sorted order, e.g. ("Fe II", "Fe III") for nnion.

    A large model has more than 5000 columns, and show_subplots reads the species of many types each time that it
    runs.
    """
    species: dict[str, set[str]] = {}
    for column in estimatorcolumns:
        if (split := split_species_suffix(column)) is not None:
            species.setdefault(split[0], set()).add(split[1])
    return MappingProxyType({
        family: tuple(sorted(names, key=get_species_sortkey)) for family, names in species.items()
    })


def get_species_choices(
    seriestype: str, estimatorcolumns: "Collection[str]", levelnames: "Sequence[str]" = ()
) -> list[str]:
    """Return each species that a type of series can plot for this model, e.g. "Fe II" for the populations.

    An ion series such as gamma_NT takes the species of its own columns, e.g. gamma_NT_Fe_II. A level population
    takes levelnames, e.g. "Fe II 0", which get_level_names gives.
    """
    if seriestype.startswith("levelpopulation"):
        return list(levelnames)
    speciesoffamilies = get_species_of_families(tuple(estimatorcolumns))
    families = SPECIES_FAMILIES.get(seriestype, (seriestype,))
    species = {name for family in families for name in speciesoffamilies.get(family, ())}
    return sorted(species, key=get_species_sortkey)


def get_first_choice(choices: "Sequence[str]") -> list[str]:
    """Return the first ion of the choices, or the first choice if no choice is an ion, e.g. Fe I before Fe."""
    ions = [choice for choice in choices if isinstance(get_iontuple(choice)[1], int)]
    return (ions or list(choices))[:1]


def is_suggested_variable(column: str) -> bool:
    """Return True if a column can be a series of its own, and not the grid, the time, or one species."""
    return (
        column not in BOOKKEEPING_COLUMNS
        and not column.startswith(("vel_", "init_pos_", "init_kinetic_"))
        and split_species_suffix(column) is None
    )


@lru_cache(maxsize=4)
def get_variable_choices(estimatorcolumns: tuple[str, ...]) -> tuple[str, ...]:
    """Return the columns that a subplot of variables can take, with the columns that suit a series first."""
    return tuple(sorted(estimatorcolumns, key=lambda column: not is_suggested_variable(column)))


@lru_cache(maxsize=4)
def get_variable_menu_groups(estimatorcolumns: tuple[str, ...]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return the groups of the menu of the variables, as pairs of a title and the columns of the group.

    The first group holds the temperatures, and the second group holds the other variables of one column. They have
    no title, thus the menu shows them at its top. Each family of columns with a shared start, e.g. heating_,
    gives a submenu. A species column goes to a subplot of its series type, thus the menu leaves it out.
    """
    temperatures: list[str] = []
    plain: list[str] = []
    families: dict[str, list[str]] = {}
    for column in sorted(estimatorcolumns, key=str.lower):
        if column in BOOKKEEPING_COLUMNS or split_species_suffix(column) is not None:
            continue
        if prefix := get_prefix_group(column):
            families.setdefault(prefix, []).append(column)
        elif get_ylabel(column).strip() == "Temperature [K]":
            temperatures.append(column)
        else:
            plain.append(column)
    return (
        ("", tuple(temperatures)),
        ("", tuple(plain)),
        *(
            (f"{PREFIX_GROUPS[prefix].capitalize()} ({prefix}…)", tuple(families[prefix]))
            for prefix in sorted(families)
        ),
    )


@lru_cache(maxsize=4)
def get_variables_of_ylabels(estimatorcolumns: tuple[str, ...]) -> "Mapping[str, tuple[str, ...]]":
    """Return the columns that suit a series of their own, by the label of their y axis."""
    variables: dict[str, list[str]] = {}
    for column in estimatorcolumns:
        if is_suggested_variable(column):
            variables.setdefault(get_ylabel(column).strip(), []).append(column)
    return MappingProxyType({ylabel: tuple(columns) for ylabel, columns in variables.items()})


def get_series_suggestions(
    subplot: "Sequence[str]", estimatorcolumns: "Collection[str]", levelnames: "Sequence[str]" = ()
) -> list[str]:
    """Return the items that suit a subplot next, e.g. TR beside Te, or Fe IV beside Fe II and Fe III.

    A series type takes more of its species, first those of the elements that the subplot has. A level population
    takes the next levels, first those of the ions that the subplot has. A variable takes the other variables with
    the same quantity on the y axis.
    """
    names = get_subplot_names(subplot)
    if not names:
        return []
    seriestype = get_subplot_seriestype(subplot, estimatorcolumns)
    if seriestype is not None and seriestype.startswith("levelpopulation"):
        ions = {name.rpartition(" ")[0] for name in names}
        choices = [name for name in levelnames if name not in names]
        return sorted(choices, key=lambda name: name.rpartition(" ")[0] not in ions)[:SERIES_SUGGESTION_COUNT]
    if seriestype is not None:
        elements = {get_iontuple(name)[0] for name in names if is_valid_ion(name)}

        # the ions of the elements of the subplot come first, and the total of an element comes after its ions
        def get_choice_sortkey(species: str) -> tuple[bool, bool, tuple[int, int, int, str]]:
            atomic_number, ion_stage = get_iontuple(species)
            return atomic_number not in elements, not isinstance(ion_stage, int), get_species_sortkey(species)

        choices = [name for name in get_species_choices(seriestype, estimatorcolumns) if name not in names]
        suggestions = sorted(choices, key=get_choice_sortkey)
    else:
        ylabel = get_ylabel(names[0]).strip()
        variables = get_variables_of_ylabels(tuple(estimatorcolumns)).get(ylabel, ()) if ylabel else ()
        suggestions = [column for column in variables if column not in names]
    return suggestions[:SERIES_SUGGESTION_COUNT]


def get_new_subplot_suggestions(
    subplots: "Sequence[Sequence[str]]",
    defaultsubplots: "Sequence[tuple[str, ...]]",
    estimatorcolumns: "Collection[str]",
) -> list[tuple[str, ...]]:
    """Return the subplots that suit the plot next.

    The list starts with the default subplots that the plot does not have. Two common variables that no subplot
    shows come next. Then come a plot of the populations and a plot of the average ionisation, if the plot has none.
    The other common variables come last.
    """
    plotted = {name for subplot in subplots for name in get_subplot_names(subplot)}
    seriestypes = {get_subplot_seriestype(subplot, estimatorcolumns) for subplot in subplots}
    variables = [(name,) for name in COMMON_VARIABLES if name in estimatorcolumns and name not in plotted]
    series: list[tuple[str, ...]] = []
    for seriestype in ("populations", "averageionisation"):
        if seriestype not in seriestypes and (choices := get_species_choices(seriestype, estimatorcolumns)):
            # the ions of the first element, e.g. Fe II and Fe III, and not the element with its first ion
            firstelement = get_iontuple(choices[0])[0]
            names = [name for name in choices if get_iontuple(name)[0] == firstelement and name != choices[0]]
            series.append((seriestype, *(names or choices)[:2]))
    # a style of a series, e.g. rho@color=C3, does not make a different subplot
    unstyled = {remove_item_styles(subplot) for subplot in subplots}
    suggestions = [
        *(subplot for subplot in defaultsubplots if subplot not in unstyled),
        *variables[:2],
        *series,
        *variables[2:],
    ]
    return list(dict.fromkeys(suggestions))[:NEW_SUBPLOT_SUGGESTION_COUNT]


def get_new_subplot(
    subplots: "Sequence[Sequence[str]]", defaultsubplots: "Sequence[tuple[str, ...]]", estimatorcolumns: "Sequence[str]"
) -> tuple[str, ...]:
    """Return the subplot that the + button of a card adds.

    plotestimators needs at least one series in each subplot. Thus the new subplot takes the first suggestion, or else
    a common variable. The card of the new subplot then sets the type and the series.
    """
    if suggestions := get_new_subplot_suggestions(subplots, defaultsubplots, estimatorcolumns):
        return suggestions[0]
    return (get_first_common_variable(estimatorcolumns),)


# the name of the type of a subplot of variables in the type selector of a subplot, which gives no -plot token
VARIABLES_TYPE: t.Final = "variables"


# the help text of each type of subplot that the selector names
SUBPLOT_TYPE_HELPTEXTS: t.Final = MappingProxyType({
    VARIABLES_TYPE: "Estimator variables with the same quantity, e.g. Te and TR",
    "populations": "The population of each ion, element, or isotope",
    "averageionisation": "The mean ion charge of each element",
    "averageexcitation": "The mean excitation energy of each ion",
    "initabundances": "The initial mass fraction of each element or isotope",
    "initmasses": "The initial mass of each element or isotope",
    "levelpopulation": "The NLTE population of each level, e.g. Fe II 0",
    "levelpopulation_dn_on_dvel": "The NLTE population of each level for each unit of velocity",
})


def get_subplot_types(estimatorcolumns: "Collection[str]", nltetypes: "Collection[str]" = ()) -> list[str]:
    """Return the types of subplot that the model can plot: the variables, the series types, and the ion series.

    An ion series is a family of columns with one column for each ion, e.g. gamma_NT_Fe_II. The families of the
    other series types, e.g. nnion for the populations, are not a type of their own. nltetypes gives the types in
    NLTE_SERIESTYPES that the model can plot, because they need its NLTE populations.
    """
    columns = tuple(estimatorcolumns)
    speciestypes = [
        seriestype
        for seriestype in SPECIES_FAMILIES
        if (seriestype not in NLTE_SERIESTYPES or seriestype in nltetypes) and get_species_choices(seriestype, columns)
    ]
    leveltypes = [seriestype for seriestype in nltetypes if seriestype.startswith("levelpopulation")]
    otherfamilies = {family for families in SPECIES_FAMILIES.values() for family in families}
    ionfamilies = [family for family in get_species_of_families(columns) if family not in otherfamilies]
    return [VARIABLES_TYPE, *speciestypes, *leveltypes, *sorted(ionfamilies, key=str.lower)]


def change_subplot_type(
    subplot: "Sequence[str]", seriestype: str, estimatorcolumns: "Collection[str]", levelnames: "Sequence[str]" = ()
) -> tuple[str, ...]:
    """Return the subplot with a new type, and keep the names and the directives that still apply.

    A new type with no name that applies takes its first choice, e.g. the first ion of the populations, because
    plotestimators needs at least one name. The new type can plot a different quantity, thus ymin= and ymax= go.
    Only a plot of populations takes ionpoptype=. If seriestype is the type of the subplot, the
    subplot stays the same.
    """
    oldtype = get_subplot_seriestype(subplot, estimatorcolumns)
    if seriestype == (oldtype or VARIABLES_TYPE):
        return tuple(subplot)
    keptdirectives = {"yscale", "ionpoptype"} if seriestype == "populations" else {"yscale"}
    directives = [item for item in subplot if get_item_directive(item) in keptdirectives]
    # a name that stays in the subplot keeps its style
    items = get_series_items(subplot)
    if oldtype is not None and items and items[0] == oldtype:
        items = items[1:]
    if seriestype == VARIABLES_TYPE:
        variables = [item for item in items if oldtype is None and get_series_name(item) in estimatorcolumns]
        return (*(variables or [get_first_common_variable(estimatorcolumns)]), *directives)
    choices = get_species_choices(seriestype, estimatorcolumns, levelnames)
    kept = [item for item in items if get_series_name(item) in choices]
    return (seriestype, *(kept or get_first_choice(choices)), *directives)


def get_moved_rows(
    oldsubplots: "Sequence[Sequence[str]]", newsubplots: "Sequence[Sequence[str]]", rows: "Collection[int]"
) -> set[int]:
    """Return the new rows of the subplots at rows, after a change of the subplots from oldsubplots to newsubplots.

    A subplot goes to the nearest row that has the same items and that no other subplot of rows took, e.g. after a
    move, an insert, a delete, or Undo. If no such row exists, the subplot keeps its row when the count of subplots
    stays the same, e.g. after a new y scale. If not, the subplot has no new row.
    """
    newrows: set[int] = set()
    for row in sorted(rows):
        if not 0 <= row < len(oldsubplots):
            continue
        matches = [
            newrow
            for newrow, subplot in enumerate(newsubplots)
            if tuple(subplot) == tuple(oldsubplots[row]) and newrow not in newrows
        ]
        if matches:
            newrows.add(min(matches, key=lambda newrow: abs(newrow - row)))
        elif len(newsubplots) == len(oldsubplots):
            newrows.add(row)
    return newrows


def get_card_summary(subplot: "Sequence[str]", estimatorcolumns: "Collection[str]") -> str:
    """Return the text that the header of a collapsed card shows in place of its controls, e.g. "Te, TR · log"."""
    items = [get_series_name(item) for _, item in get_chip_items(subplot, estimatorcolumns)]
    maxshown = 4
    text = ", ".join(items[:maxshown]) + (f" +{len(items) - maxshown}" if len(items) > maxshown else "")
    yscale = get_directive_value(subplot, "yscale")
    return f"{text} · {yscale}" if yscale else text


def get_chip_items(subplot: "Sequence[str]", estimatorcolumns: "Collection[str]") -> list[tuple[int, str]]:
    """Return the position and the item of each series of a subplot that shows as a chip, with the style of the item.

    The type selector shows the series type, and a control of the card sets each directive, thus they show no chip.
    """
    seriestype = get_subplot_seriestype(subplot, estimatorcolumns)
    typeposition = next((position for position, item in enumerate(subplot) if item == seriestype), None)
    return [
        (position, item)
        for position, item in enumerate(subplot)
        if position != typeposition and get_item_directive(item) is None
    ]


# the quantity of the ions of a populations subplot with no ionpoptype=, which is the default of -ionpoptype
DEFAULT_POPTYPE: t.Final = "absolute"


def move_poptype_to_subplots(
    subplots: "Sequence[tuple[str, ...]]", otheroptions: OptionRows, estimatorcolumns: "Collection[str]"
) -> tuple[tuple[tuple[str, ...], ...], OptionRows]:
    """Return the subplots and the rows of the option table with --ionpoptype in each populations subplot.

    The window sets the quantity of the ions for each populations subplot (ionpoptype=). Thus --ionpoptype of the
    command goes to each populations subplot that has no ionpoptype= of its own. With no populations subplot, the row
    stays, and a populations subplot that the user adds later takes it.
    """
    poptype = next((values[0] for flag, values in otheroptions if flag == "--ionpoptype" and values), None)
    ispopulations = [get_subplot_seriestype(subplot, estimatorcolumns) == "populations" for subplot in subplots]
    if poptype is None or not any(ispopulations):
        return tuple(subplots), otheroptions
    rows = tuple((flag, values) for flag, values in otheroptions if flag != "--ionpoptype")
    return tuple(
        (*subplot, f"ionpoptype={poptype}")
        if populations and poptype != DEFAULT_POPTYPE and get_directive_value(subplot, "ionpoptype") is None
        else subplot
        for subplot, populations in zip(subplots, ispopulations, strict=True)
    ), rows


def remove_subplot_item(subplots: "Sequence[tuple[str, ...]]", row: int, index: int) -> tuple[tuple[str, ...], ...]:
    """Return the subplots without one item of a subplot.

    A subplot with no series left stays, and it draws an empty frame until the user adds a series. Only the ✕ of
    the card deletes a subplot.
    """
    subplot = subplots[row][:index] + subplots[row][index + 1 :]
    return (*subplots[:row], subplot, *subplots[row + 1 :])


def move_series_item(
    subplots: "Sequence[tuple[str, ...]]",
    source: tuple[int, int],
    targetrow: int,
    chipindex: int,
    estimatorcolumns: "Collection[str]",
) -> tuple[tuple[str, ...], ...] | None:
    """Return the subplots with a series item moved in front of a chip of a subplot, or None for a repeated name.

    source gives the row of the subplot and the position of the item in it. chipindex is the index of a chip of the
    target subplot on the screen, see get_chip_items. An index equal to the number of chips puts the item after the
    last chip. The type and the directives of the target subplot stay at their places. The item keeps its style.
    """
    row, position = source
    item = subplots[row][position]
    if targetrow != row and get_series_name(item) in get_subplot_names(subplots[targetrow]):
        return None
    if targetrow == row:
        # the chips on the screen still show the moved item, thus a later chip moves one place to the front
        oldchips = [chipposition for chipposition, _ in get_chip_items(subplots[row], estimatorcolumns)]
        if position in oldchips and chipindex > oldchips.index(position):
            chipindex -= 1
    removed = remove_subplot_item(subplots, row, position)
    target = removed[targetrow]
    chips = get_chip_items(target, estimatorcolumns)
    if chipindex < len(chips):
        insertat = chips[chipindex][0]
    elif chips:
        insertat = chips[-1][0] + 1
    else:
        # a subplot with no chip holds its type or its directives alone, and a name goes after the type
        insertat = next(
            (index for index, other in enumerate(target) if get_item_directive(other) is not None), len(target)
        )
    newtarget = (*target[:insertat], item, *target[insertat:])
    return (*removed[:targetrow], newtarget, *removed[targetrow + 1 :])


def get_command_items(subplot: "Sequence[str]", estimatorcolumns: "Collection[str]") -> tuple[str, ...]:
    """Return the items of a subplot that the command gives.

    A series type with no names has nothing to plot, and plotestimators rejects it. The card keeps the type, and the
    command gives the directives alone, which draw an empty frame.
    """
    names = get_subplot_names(subplot)
    seriestypeonly = (
        len(names) == 1
        and names[0] not in estimatorcolumns
        and (is_seriestype(names[0], estimatorcolumns) or bool(get_species_choices(names[0], estimatorcolumns)))
    )
    return tuple(item for item in subplot if get_item_directive(item) is not None) if seriestypeonly else tuple(subplot)


def get_card_key(
    row: int, subplots: "Sequence[Sequence[str]]", estimatorcolumns: "Sequence[str]", *, isimage: bool = False
) -> tuple[object, ...]:
    """Return the parts of the plot that the widgets of the card of a subplot show.

    A directive, e.g. ymin=, changes only the text of a control of the card, thus the card stays. The focus then stays
    in the control, e.g. in the field of the y maximum after an edit of the y minimum. A chip removes the
    item at its position, thus the key holds the position of each name. A style of a series, e.g. @color=C3, also
    changes only the controls, thus the key holds the names without the styles. A colour image gives the card the
    controls of a colour scale, thus the key also holds isimage.
    """
    names = tuple(
        (position, get_series_name(item))
        for position, item in enumerate(subplots[row])
        if get_item_directive(item) is None
    )
    return (row, row == len(subplots) - 1, names, estimatorcolumns, isimage)


class SubplotCard(t.NamedTuple):
    """The card of a subplot in the window, with the controls that show the directives of the subplot."""

    key: tuple[object, ...]
    frame: "QtWidgets.QFrame"
    yscalebox: "QtWidgets.QComboBox"
    poptypebox: "QtWidgets.QComboBox | None"
    yminedit: "QtWidgets.QLineEdit"
    ymaxedit: "QtWidgets.QLineEdit"
    # the button of the line of each series chip, by the name of the series. A plot gives each button its image
    swatchbuttons: "dict[str, QtWidgets.QToolButton]"
    # the widget of the chips, and the position in the subplot and the widget of each chip, for a drag of a chip
    chipsbox: "QtWidgets.QWidget"
    chips: "list[tuple[int, QtWidgets.QWidget]]"
    # the button that shows a summary of the controls of the y axis, and opens or closes the box of the controls
    yaxisbutton: "QtWidgets.QToolButton"
    yaxisbox: "QtWidgets.QWidget"
    # a note for a subplot of ions with more than one element, which keeps the element colours with --colorbyion
    colournote: "QtWidgets.QLabel"
    # the item of the menu of the header that removes the styles, which a subplot with no style disables
    resetaction: "QtGui.QAction"


def get_subplot_style_sheet() -> str:
    """Return the style of the cards, the chips, and the suggestions, in the accent colour of the system.

    A card is a grouped box with a light fill and no border, as in the settings of macOS. A chip is a token of the
    address field of Mail: a pill with a light tint of the accent colour. A suggestion is plain text in the accent
    colour. A style sheet for each widget took about 10 ms each time that the window showed the cards. One style sheet
    for all the cards takes less time.
    """
    accent = get_accent_colour()
    rgb = f"{accent.red()}, {accent.green()}, {accent.blue()}"
    textaccent = get_accent_colour(fortext=True)
    textrgb = f"{textaccent.red()}, {textaccent.green()}, {textaccent.blue()}"
    # the alpha of the tint suits the light and the dark appearance
    return (
        "QFrame#subplotcard { border: none; border-radius: 8px; background: rgba(128, 128, 128, 28); }"
        f" #chip {{ border: none; border-radius: 10px; background: rgba({rgb}, 46); }}"
        f" #chip:focus {{ border: 2px solid rgb({rgb}); }}"
        " QToolButton#swatch { border: none; padding: 2px 3px; border-radius: 4px; background: transparent; }"
        " QToolButton#swatch:hover { background: rgba(128, 128, 128, 60); }"
        " QToolButton#yaxis { border: none; padding: 1px 0px; background: transparent; }"
        f" QToolButton#suggestion {{ border: none; border-radius: 10px; padding: 1px 6px; color: rgb({textrgb});"
        " background: transparent; }"
        f" QToolButton#suggestion:hover {{ background: rgba({rgb}, 46); }}"
        f" QFrame#dropline {{ background: rgb({rgb}); border: none; }}"
        f" QWidget#dragheader:focus {{ border: 2px solid rgb({rgb}); border-radius: 4px; }}"
    )


# the width in pixels of the image of the line of a chip. A shorter line than in the list of the series keeps the name
# of the series the main part of the chip
CHIP_SWATCH_WIDTH: t.Final = 22


def make_chip(
    chip: "QtWidgets.QWidget",
    text: str,
    tooltip: str,
    on_remove: "Callable[[], None]",
    swatchbutton: "QtWidgets.QToolButton | None",
) -> None:
    """Fill a chip that shows one item of a subplot, with a button that removes the item.

    chip comes from make_drag_header, thus the user can drag it, move it with Alt-Up and Alt-Down, and click it.
    swatchbutton shows the line of the series at the start of the chip, and a colour image gives None.
    show_card_directives puts a dot after the text of a series that has a style of its own. get_subplot_style_sheet
    gives the chip its tint.
    """
    from PySide6 import QtWidgets

    chip.setObjectName("chip")
    layout = QtWidgets.QHBoxLayout(chip)
    # the padding of the line button and the margin give the same space at the start of a chip as the ✕ at its end
    layout.setContentsMargins(4 if swatchbutton is not None else 8, 1, 2, 1)
    layout.setSpacing(0)
    if swatchbutton is not None:
        layout.addWidget(swatchbutton)
        # the line ends at the edge of its image, thus the name needs a gap to read as a separate item
        layout.addSpacing(4)
    label = QtWidgets.QLabel(text)
    label.setObjectName("chiplabel")
    label.setToolTip(tooltip)
    removebutton = make_glyph_button("xmark", f"Remove {text} from the subplot", f"Remove {text}")
    removebutton.clicked.connect(on_remove)
    layout.addWidget(label)
    layout.addWidget(removebutton)


def set_drag_fade(widget: "QtWidgets.QWidget", *, faded: bool) -> None:
    """Fade a widget that the user drags, e.g. a card or a chip, or show it as before."""
    from PySide6 import QtWidgets

    if (fade := widget.graphicsEffect()) is None:
        fade = QtWidgets.QGraphicsOpacityEffect(widget)
        fade.setOpacity(0.5)
        widget.setGraphicsEffect(fade)
    fade.setEnabled(faded)


def get_linewidthscale(parser: argparse.ArgumentParser, rows: OptionRows) -> float:
    """Return the --linewidthscale of the option rows, or the default for no row or a value that is not a number.

    The rows keep a value that the worker rejects until the plot ends.
    """
    default = float(parser.get_default("linewidthscale"))
    try:
        return float((get_row_values(rows, "--linewidthscale") or (default,))[0])
    except ValueError:
        return default


# the functions of the section of the subplots that the window calls, see add_subplot_section
type SubplotSectionFunctions = tuple[
    Callable[[], None],
    Callable[[int, int], None],
    Callable[[int, Mapping[str, str | None]], None],
    Callable[[], None],
    Callable[[], None],
]


def add_subplot_section(
    panellayout: "QtWidgets.QVBoxLayout",
    window: "QtWidgets.QMainWindow",
    viewer: "EstimatorViewer",
    get_queue: "Callable[[], DrawQueue[ControlValues]]",
    plot_shows_values: "Callable[[], bool]",
) -> SubplotSectionFunctions:
    """Add the section of the subplots to the panel of the window, with a card for each subplot.

    get_queue returns the queue of the plots of the window. plot_shows_values tells if the plot on the screen has the
    values of the viewer. The function returns the functions that the window calls:

    - show_subplot_section shows the subplots of the viewer;
    - edit_item_style asks for the line properties of a series, e.g. after a click on its line;
    - set_directives changes the directives of a subplot, e.g. after a Shift-drag;
    - on_add_series_key and on_add_subplot_key give the actions Add Series and Add Subplot of the menu.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    # interactive imports this module, thus the names of interactive come when the window opens
    from artistools.estimators.interactive import get_geometry_mode
    from artistools.estimators.interactive import get_plot_frames
    from artistools.estimators.interactive import IMAGE_MODES

    helptexts = viewer.helptexts

    def apply(values: "ControlValues", *, undoable: bool = True) -> None:
        get_queue().apply(values, undoable=undoable)

    def show_error(message: str) -> None:
        get_queue().show_error(message)

    _, subplotgrid = add_section(panellayout, "Subplots")
    # show_subplots makes the card of a subplot again when the names of the subplot change
    subplotsbox = QtWidgets.QWidget()
    subplotstylesheet = get_subplot_style_sheet()
    subplotsbox.setStyleSheet(subplotstylesheet)
    # a new appearance gives a new accent colour, see refresh_palette_style_sheets
    subplotsbox.setProperty("stylesheetfactory", get_subplot_style_sheet)
    subplotslayout = QtWidgets.QVBoxLayout(subplotsbox)
    subplotslayout.setContentsMargins(0, 0, 0, 0)
    subplotslayout.setSpacing(6)
    # the rows of the cards that show only their header
    collapsedrows: set[int] = set()
    # the rows of the cards that show the controls of the y axis
    openyrows: set[int] = set()
    # the subplots of the cards on the screen. After each change, e.g. a move, Undo, or a rejected change,
    # show_subplots moves the collapsed cards with their subplots
    shownsubplots: tuple[tuple[str, ...], ...] = ()
    # the line that shows where a dragged card goes
    dropline = QtWidgets.QFrame(subplotsbox)
    dropline.setObjectName("dropline")
    dropline.hide()
    # the vertical line that shows where a dragged chip goes
    chipdropline = QtWidgets.QFrame(subplotsbox)
    chipdropline.setObjectName("dropline")
    chipdropline.hide()
    defaultbutton = QtWidgets.QPushButton("Default")
    defaultbutton.setToolTip("Show the default subplots of plotestimators for this model")
    newsuggestionsbox = QtWidgets.QWidget()
    newsuggestionsbox.setStyleSheet(subplotstylesheet)
    newsuggestionsbox.setProperty("stylesheetfactory", get_subplot_style_sheet)
    newsuggestionslayout = make_flow_layout()
    newsuggestionsbox.setLayout(newsuggestionslayout)
    subplotgrid.addWidget(subplotsbox, 0, 0, 1, -1)
    subplotgrid.addWidget(defaultbutton, 1, 0, 1, -1, QtCore.Qt.AlignmentFlag.AlignLeft)
    subplotgrid.addWidget(newsuggestionsbox, 2, 0, 1, -1)
    skippeddefaultslabel = make_note_label()
    subplotgrid.addWidget(skippeddefaultslabel, 3, 0, 1, -1)

    # the cards of the subplots on the screen, and the key of the suggestions of a new subplot
    cards: list[SubplotCard] = []
    shownnewkey: tuple[object, ...] = ()
    # the row of a card, the name of its control, and the position of a chip, which take the focus when show_subplots
    # runs next. Each chip has the name "chip", thus the position selects one
    pendingfocus: tuple[int, str, int | None] | None = None

    def make_suggestion_button(text: str, tooltip: str, callback: "Callable[[], None]") -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton()
        button.setObjectName("suggestion")
        button.setText(f"+ {text}")
        button.setToolTip(tooltip)
        button.clicked.connect(callback)
        return button

    def set_box_text(box: QtWidgets.QComboBox, text: str) -> None:
        # a directive of the command can give a value that is not a choice, e.g. yscale=symlog
        if box.findText(text) < 0:
            box.addItem(text)
        box.setCurrentText(text)

    def make_selector(
        name: str, choices: "Sequence[str]", current: str, tooltip: str, callback: "Callable[[str], None]"
    ) -> QtWidgets.QComboBox:
        box = QtWidgets.QComboBox()
        box.setObjectName(name)
        box.addItems(list(choices))
        set_box_text(box, current)
        box.setToolTip(tooltip)
        box.textActivated.connect(callback)
        return box

    def get_yscale_choice(subplot: "Sequence[str]") -> str:
        value = get_directive_value(subplot, "yscale") or "auto"
        return {"lin": "linear"}.get(value, value)

    def make_subplot_card(
        row: int, subplot: tuple[str, ...], subplottypes: "Sequence[str]", key: tuple[object, ...], *, isimage: bool
    ) -> SubplotCard:
        """Return the card of a subplot. The controls of the card follow the type of the subplot.

        The name of the object of a control gives its role. show_subplots gives the focus to the control with the same
        role in a new card.
        """
        columns = viewer.modeldata.estimatorcolumns
        seriestype = get_subplot_seriestype(subplot, columns)
        currenttype = seriestype or VARIABLES_TYPE
        names = get_subplot_names(subplot)
        card = QtWidgets.QFrame()
        card.setObjectName("subplotcard")
        framelayout = QtWidgets.QVBoxLayout(card)
        framelayout.setContentsMargins(6, 4, 4, 6)
        framelayout.setSpacing(4)

        typebox = QtWidgets.QComboBox()
        typebox.setObjectName("type")
        types = [*subplottypes, *([] if currenttype in subplottypes else [currenttype])]
        typebox.addItems(types)
        for index, name in enumerate(types):
            helptext = SUBPLOT_TYPE_HELPTEXTS.get(name, f"The {name} columns of each ion")
            typebox.setItemData(index, helptext, QtCore.Qt.ItemDataRole.ToolTipRole)
        typebox.setCurrentText(currenttype)
        typebox.setToolTip("The type of the subplot. A new type keeps the names that still apply.")
        typebox.textActivated.connect(partial(on_subplot_type, row))
        iscollapsed = row in collapsedrows
        quantity = QtWidgets.QLabel(
            get_card_summary(subplot, columns)
            if iscollapsed
            else plain_label(get_ylabel(names[0])).strip()
            if seriestype is None and names
            else ""
        )
        quantity.setEnabled(False)
        # a narrow sidebar cuts the text of the header and the type, and the buttons of the header stay in view
        quantity.setMinimumWidth(1)
        typebox.setMinimumWidth(100)
        disclosure = QtWidgets.QToolButton()
        set_disclosure_symbol(disclosure, isopen=not iscollapsed)
        disclosure.setStyleSheet("QToolButton { border: none; }")
        disclosure.setToolTip("Show the controls of the subplot" if iscollapsed else "Hide the controls of the subplot")
        disclosure.setAccessibleName("Expand" if iscollapsed else "Collapse")
        disclosure.clicked.connect(partial(on_collapse_subplot, row))
        header = make_drag_header(
            partial(on_drag_subplot, row), partial(on_drop_subplot, row), partial(on_move_key, row)
        )
        header.setToolTip("Drag the header to move the subplot. Alt-Up and Alt-Down (Option on a Mac) also move it.")
        headerlayout = QtWidgets.QHBoxLayout(header)
        headerlayout.setContentsMargins(0, 0, 0, 0)
        headerlayout.addWidget(disclosure)
        headerlayout.addWidget(QtWidgets.QLabel(f"<b>{row + 1}</b>"))
        headerlayout.addWidget(typebox)
        headerlayout.addWidget(quantity, 1)
        # the plot needs at least one subplot. The key of the card changes with the last row, thus the state stays true
        isonlysubplot = len(viewer.values.subplots) == 1
        for symbol, tooltip, enabled, callback in (
            ("plus", "Insert a new subplot below this subplot", True, partial(insert_new_subplot, row + 1)),
            ("xmark", "Delete the subplot", not isonlysubplot, partial(on_delete_subplot, row)),
        ):
            button = make_glyph_button(symbol, tooltip, tooltip)
            button.setEnabled(enabled)
            if not enabled:
                button.setToolTip("The plot needs at least one subplot")
            button.clicked.connect(callback)
            headerlayout.addWidget(button)
        headerlayout.addWidget(make_grip("Drag the header to move the subplot"))
        # the keyboard and VoiceOver reach the moves through the menu of the header
        header.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.ActionsContextMenu)
        for text, target, enabled in (
            ("Move Up", row - 1, row > 0),
            ("Move Down", row + 1, row < len(viewer.values.subplots) - 1),
        ):
            action = QtGui.QAction(text, header)
            action.setEnabled(enabled)
            action.triggered.connect(partial(move_subplot, row, target))
            header.addAction(action)
        resetaction = QtGui.QAction("Reset Line Properties", header)
        resetaction.triggered.connect(partial(on_reset_styles, row))
        header.addAction(resetaction)
        framelayout.addWidget(header)
        body = QtWidgets.QWidget()
        cardlayout = QtWidgets.QVBoxLayout(body)
        cardlayout.setContentsMargins(0, 0, 0, 0)
        cardlayout.setSpacing(6)
        framelayout.addWidget(body)
        # a collapsed card shows only its header. A widget with no parent shows as a window, which closes an open popup.
        # Thus the body goes into the card before setVisible
        body.setVisible(not iscollapsed)

        chipsbox = QtWidgets.QWidget()
        chipslayout = make_flow_layout()
        chipsbox.setLayout(chipslayout)
        swatchbuttons: dict[str, QtWidgets.QToolButton] = {}
        chips: list[tuple[int, QtWidgets.QWidget]] = []
        for position, item in get_chip_items(subplot, columns):
            name = get_series_name(item)
            if seriestype is None:
                tooltip = plain_label(get_ylabel(name)).strip() or name
            else:
                tooltip = f"The {currenttype} of {name}"
            tooltip += (
                ". Drag the chip to move the series."
                if isimage
                else ". Click to change the line. Drag the chip to move the series."
            )
            # a colour image has no lines, thus its chips have no line style
            swatchbutton = None
            if not isimage:
                swatchbutton = make_swatch_button(name, CHIP_SWATCH_WIDTH, partial(on_click_chip, row, position))
                swatchbuttons[name] = swatchbutton
            chip = make_drag_header(
                partial(on_drag_chip, row, position),
                partial(on_drop_chip, row, position),
                partial(on_move_chip, row, position),
                None if isimage else partial(on_click_chip, row, position),
            )
            make_chip(chip, name, tooltip, partial(on_remove_item, row, position), swatchbutton)
            chip.setProperty("chipposition", position)
            chipslayout.addWidget(chip)
            chips.append((position, chip))
        cardlayout.addWidget(chipsbox)
        colournote = make_note_label()
        # set_note_text shows the label, and a label with no parent shows as a window of its own
        cardlayout.addWidget(colournote)
        set_note_text(
            colournote,
            "Each ion takes the colour of its element, because the subplot holds more than one element. Each ion stage"
            " has a different dash pattern.",
        )
        colournote.setToolTip(helptexts.get("colorbyion", ""))
        colournote.setVisible(row in viewer.elementcolourrows)

        levelnames = get_levelnames(currenttype)
        choices = (
            get_species_choices(currenttype, columns, levelnames)
            if seriestype is not None
            else get_variable_choices(tuple(columns))
        )
        suggestions = get_series_suggestions(subplot, columns, levelnames)
        example = f", e.g. {suggestions[0]}" if suggestions else ""
        addedit = QtWidgets.QLineEdit()
        addedit.setObjectName("add")
        addcompleter = make_completer(choices, addedit)
        addedit.setCompleter(addcompleter)
        # the field takes the name first, and then this handler adds it
        addcompleter.activated.connect(partial(on_complete_item, row, addedit))
        if currenttype == VARIABLES_TYPE:
            addedit.setPlaceholderText(f"Add a variable{example}")
        elif currenttype.startswith("levelpopulation"):
            addedit.setPlaceholderText(f"Add a level: an ion and the index of its level{example}")
        elif currenttype == "populations":
            addedit.setPlaceholderText(f"Add an ion, an element, or an isotope{example}")
        else:
            addedit.setPlaceholderText(f"Add a species{example}")
        addedit.setToolTip(
            "Press Return to add the name. A part of a name shows the names that hold it. A directive such as"
            " ymin=1e-16 also goes here."
        )
        addedit.returnPressed.connect(partial(on_add_item, row, addedit))
        listbutton = make_glyph_button("chevron.down", "Show each name that the subplot can take", "Show All Names")
        if currenttype == VARIABLES_TYPE:
            listbutton.clicked.connect(partial(show_variable_menu, row, listbutton))
        else:
            listbutton.clicked.connect(partial(show_all_choices, addedit))
        addrow = QtWidgets.QHBoxLayout()
        addrow.addWidget(addedit, 1)
        addrow.addWidget(listbutton)
        cardlayout.addLayout(addrow)

        if suggestions:
            suggestionsbox = QtWidgets.QWidget()
            suggestionslayout = make_flow_layout()
            suggestionsbox.setLayout(suggestionslayout)
            for suggestion in suggestions:
                suggestionslayout.addWidget(
                    make_suggestion_button(
                        suggestion, f"Add {suggestion} to the subplot", partial(add_item, row, suggestion)
                    )
                )
            cardlayout.addWidget(suggestionsbox)

        # a colour image shows the values as colours, and its vertical axis is a velocity. Thus the directives of the
        # y axis set the colour scale
        yscalebox = make_selector(
            "yscale",
            ("auto", "linear", "log"),
            get_yscale_choice(subplot),
            "The colour scale of the values (yscale=). Auto takes log for values that cover many decades."
            if isimage
            else "The scale of the y axis (yscale=). Auto takes log for ions and linear for the other series.",
            partial(on_directive_selector, row, "yscale", "auto"),
        )
        poptypebox = None
        if currenttype == "populations":
            poptypebox = make_selector(
                "poptype",
                tuple(POPTYPE_YLABELS),
                get_directive_value(subplot, "ionpoptype") or DEFAULT_POPTYPE,
                f"The quantity of each ion of this subplot (ionpoptype=). {DEFAULT_POPTYPE} needs no directive.",
                partial(on_directive_selector, row, "ionpoptype", DEFAULT_POPTYPE),
            )
            cardlayout.addLayout(make_row_layout([QtWidgets.QLabel("Quantity:"), poptypebox]))

        yminedit, ymaxedit = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        for edit, directive in ((yminedit, "ymin"), (ymaxedit, "ymax")):
            edit.setObjectName(directive)
            edit.setFixedWidth(70)
            edit.setPlaceholderText("auto")
            edit.setText(get_directive_value(subplot, directive) or "")
            extent = f"{directive[1:]}imum"
            edit.setToolTip(
                f"The {extent} of the colour scale ({directive}=)."
                if isimage
                else f"The {extent} of the y axis ({directive}=). Shift-drag on the subplot sets it."
            )
            edit.editingFinished.connect(partial(on_yrange, row, yminedit, ymaxedit))
        setrangebutton = QtWidgets.QPushButton("Current")
        setrangebutton.setToolTip(
            "Set the value min and max to the current range of the colour scale. The colours then keep their"
            " meaning at each timestep."
            if isimage
            else "Set the y min and max to the current range of the y axis. The axis then stays the same at each"
            " timestep."
        )
        setrangebutton.setAccessibleName("Set Current Range")
        setrangebutton.clicked.connect(partial(on_set_current_range, row))
        autorangebutton = QtWidgets.QPushButton("Auto")
        autorangebutton.setToolTip(
            "Remove the value min and max, thus the colour scale follows the values of each timestep."
            if isimage
            else "Remove the y min and max, thus the y axis follows the data of each timestep."
        )
        autorangebutton.setAccessibleName("Automatic Range")
        autorangebutton.clicked.connect(partial(set_directives, row, {"ymin": None, "ymax": None}))
        quantityname = "value" if isimage else "y"
        # the controls of the axis take a row that most subplots do not need, thus a summary shows them on a click
        yaxisbutton = QtWidgets.QToolButton()
        yaxisbutton.setObjectName("yaxis")
        yaxisbutton.setAutoRaise(True)
        # the keyboard of macOS reaches only the controls with a strong focus, and the y range has no other control
        yaxisbutton.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        yaxisbutton.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        isyaxisopen = row in openyrows
        set_disclosure_symbol(yaxisbutton, isopen=isyaxisopen)
        yaxisbutton.setToolTip(f"Show or hide the scale and the range of the {'values' if isimage else 'y axis'}")
        yaxisbutton.clicked.connect(partial(on_yaxis_toggle, row))
        yaxisbox = QtWidgets.QWidget()
        yaxisbox.setLayout(
            make_row_layout([
                QtWidgets.QLabel(f"{quantityname} scale:"),
                yscalebox,
                QtWidgets.QLabel("min:"),
                yminedit,
                QtWidgets.QLabel("max:"),
                ymaxedit,
                setrangebutton,
                autorangebutton,
            ])
        )
        cardlayout.addWidget(yaxisbutton)
        cardlayout.addWidget(yaxisbox)
        yaxisbox.setVisible(isyaxisopen)
        return SubplotCard(
            key=key,
            frame=card,
            yscalebox=yscalebox,
            poptypebox=poptypebox,
            yminedit=yminedit,
            ymaxedit=ymaxedit,
            swatchbuttons=swatchbuttons,
            chipsbox=chipsbox,
            chips=chips,
            yaxisbutton=yaxisbutton,
            yaxisbox=yaxisbox,
            colournote=colournote,
            resetaction=resetaction,
        )

    # the names of the NLTE levels of the model, which get_levelnames reads when a card first needs them
    levelnamescache: list[list[str]] = []

    def get_levelnames(seriestype: str) -> list[str]:
        """Return the names of the levels for a level population, or no names for another type of series.

        If get_level_names cannot read the NLTE populations, the terminal shows the error, and the level population
        offers no level.
        """
        if not seriestype.startswith("levelpopulation") or not viewer.modeldata.cells:
            return []
        if not levelnamescache:
            names: list[str] = []
            timestep, cell = viewer.values.last, viewer.modeldata.cells[0]
            run_command_step(lambda: names.extend(get_level_names(viewer.modelpath, timestep, cell)))
            levelnamescache.append(names)
        return levelnamescache[0]

    def show_variable_menu(row: int, button: QtWidgets.QToolButton) -> None:
        """Show a menu of the variables in groups, with the units of each variable at the right.

        A variable that the subplot shows has a check mark, and the menu does not offer it again.
        """
        names = set(get_subplot_names(viewer.values.subplots[row]))
        # a plot that ends while the menu shows can make the card again, and that deletes the children of the card
        menu = QtWidgets.QMenu(window)
        for title, columns in get_variable_menu_groups(tuple(viewer.modeldata.estimatorcolumns)):
            if not columns:
                continue
            # a line separates the groups at the top, and the submenus stay together below them
            if not menu.isEmpty() and not (title and menu.actions()[-1].menu() is not None):
                menu.addSeparator()
            target = menu.addMenu(title) if title else menu
            for column in columns:
                # the text after a tab goes to the right edge of the menu, as a shortcut does
                action = target.addAction(f"{column}\t{format_units(column).strip()}")
                action.setCheckable(True)
                action.setChecked(column in names)
                action.setEnabled(column not in names)
                action.triggered.connect(partial(add_item, row, column))
        menu.exec(button.mapToGlobal(button.rect().bottomLeft()))
        menu.deleteLater()

    def show_all_choices(edit: QtWidgets.QLineEdit) -> None:
        if (completer := edit.completer()) is not None:
            edit.setFocus()
            completer.setCompletionPrefix(edit.text())
            completer.complete()

    def remove_widget(layout: QtWidgets.QLayout, widget: QtWidgets.QWidget) -> None:
        """Remove a widget from the window.

        A field that loses the focus sends editingFinished, thus the controls of the widget stop their signals first.
        """
        for child in widget.findChildren(QtWidgets.QWidget):
            child.blockSignals(True)  # ruff:ignore[boolean-positional-value-in-call]
        layout.removeWidget(widget)
        widget.hide()
        widget.deleteLater()

    def show_card_directives(row: int, card: SubplotCard, subplot: "Sequence[str]") -> None:
        """Show the directives of a subplot on the controls of its card, e.g. the ymin= of a Shift-drag."""
        set_box_text(card.yscalebox, get_yscale_choice(subplot))
        if card.poptypebox is not None:
            set_box_text(card.poptypebox, get_directive_value(subplot, "ionpoptype") or DEFAULT_POPTYPE)
        ymin, ymax = get_directive_value(subplot, "ymin"), get_directive_value(subplot, "ymax")
        set_edit_text(card.yminedit, ymin or "")
        set_edit_text(card.ymaxedit, ymax or "")
        axisname = "Values" if get_geometry_mode(viewer.values) in IMAGE_MODES else "y axis"
        yrange = "auto range" if ymin is None and ymax is None else f"{ymin or 'auto'} to {ymax or 'auto'}"
        card.yaxisbutton.setText(f"{axisname}: {get_yscale_choice(subplot)} scale, {yrange}")
        card.colournote.setVisible(row in viewer.elementcolourrows)
        # a dot after the name of a chip shows that its series has a style of its own
        for position, chip in card.chips:
            if position < len(subplot) and (label := chip.findChild(QtWidgets.QLabel, "chiplabel")) is not None:
                name = get_series_name(subplot[position])
                label.setText(f"{name} •" if get_item_style(subplot[position]) else name)
        card.resetaction.setEnabled(remove_item_styles(subplot) != tuple(subplot))

    def show_new_subplot_suggestions() -> None:
        """Show the suggestions of a new subplot."""
        oldbuttons = [
            item.widget()
            for index in range(newsuggestionslayout.count())
            if (item := newsuggestionslayout.itemAt(index)) is not None
        ]
        for button in oldbuttons:
            if button is not None:
                remove_widget(newsuggestionslayout, button)
        suggestions = get_new_subplot_suggestions(
            viewer.values.subplots, viewer.modeldata.defaultsubplots, viewer.modeldata.estimatorcolumns
        )
        for subplot in suggestions:
            text = shlex.join(subplot)
            newsuggestionslayout.addWidget(
                make_suggestion_button(text, f"Add a subplot of {text}", partial(add_suggested_subplot, subplot))
            )
        newsuggestionsbox.setVisible(bool(suggestions))

    def show_subplots() -> None:
        """Show a card for each subplot, and the suggestions of a new subplot.

        A card with the same key stays, and its controls show the new directives. A new card of the same row gives
        the focus to the control that had it in the old card.
        """
        nonlocal shownnewkey, pendingfocus, shownsubplots
        subplots, columns = viewer.values.subplots, viewer.modeldata.estimatorcolumns
        if subplots != shownsubplots:
            for rows in (collapsedrows, openyrows):
                movedrows = get_moved_rows(shownsubplots, subplots, rows)
                rows.clear()
                rows.update(movedrows)
            shownsubplots = subplots
        subplottypes = get_subplot_types(columns, viewer.nltetypes)
        isimage = get_geometry_mode(viewer.values) in IMAGE_MODES
        focuswidget = QtWidgets.QApplication.focusWidget()
        focus, pendingfocus = pendingfocus, None
        for row, subplot in enumerate(subplots):
            # the header of a collapsed card shows the summary, which holds the y scale
            summary = get_card_summary(subplot, columns) if row in collapsedrows else None
            key = (*get_card_key(row, subplots, columns, isimage=isimage), summary)
            if row < len(cards) and cards[row].key == key:
                show_card_directives(row, cards[row], subplot)
                continue
            if row < len(cards):
                oldframe = cards[row].frame
                if focus is None and focuswidget is not None and oldframe.isAncestorOf(focuswidget):
                    chipposition = focuswidget.property("chipposition")
                    focus = (row, focuswidget.objectName(), chipposition if isinstance(chipposition, int) else None)
                remove_widget(subplotslayout, oldframe)
            card = make_subplot_card(row, subplot, subplottypes, key, isimage=isimage)
            show_card_directives(row, card, subplot)
            subplotslayout.insertWidget(row, card.frame)
            if row < len(cards):
                cards[row] = card
            else:
                cards.append(card)
        for card in cards[len(subplots) :]:
            remove_widget(subplotslayout, card.frame)
        del cards[len(subplots) :]
        show_swatches()
        newkey = (subplots, viewer.modeldata.defaultsubplots, columns)
        if newkey != shownnewkey:
            show_new_subplot_suggestions()
            shownnewkey = newkey
        if focus is not None and focus[1] and focus[0] < len(cards):
            row, name, chipposition = focus
            target = (
                get_chip_widget(row, chipposition)
                if chipposition is not None
                else cards[row].frame.findChild(QtWidgets.QWidget, name)
            )
            if target is not None:
                # a popup of a completer gives the focus back when it hides, thus the control takes it after the popup
                QtCore.QTimer.singleShot(0, target, target.setFocus)

    def show_swatches() -> None:
        """Show the line of each series of the last plot on the button of its chip.

        A series that the plot does not show, e.g. an ion that the model does not hold, shows no line.
        """
        for row, card in enumerate(cards):
            for name, button in card.swatchbuttons.items():
                swatch = viewer.swatches.get((row, name))
                # a new icon lays out the chips again, thus a line that did not change keeps its icon
                if button.property("swatch") == swatch:
                    continue
                button.setProperty("swatch", swatch)
                button.setIcon(
                    QtGui.QIcon(
                        make_line_swatch(
                            swatch.colour, swatch.alpha, swatch.linewidth, swatch.dashpattern, CHIP_SWATCH_WIDTH
                        )
                    )
                    if swatch is not None
                    else QtGui.QIcon()
                )

    def edit_item_style(row: int, position: int, anchor: QtCore.QPoint | None = None) -> None:
        """Ask for the line properties of one series of a subplot, and show each change in the plot immediately."""
        subplot = viewer.values.subplots[row]
        if position >= len(subplot):
            return
        item = subplot[position]
        name = get_series_name(item)
        style = get_item_style(item)
        # a series that the plot does not show, e.g. an ion that the model does not hold, has no line
        swatch = viewer.swatches.get((row, name))

        def show_changes(changes: "Mapping[str, str | None] | None", undoable: bool) -> None:
            # a change applies to the current values, because Play can move the time while the popup is open. A
            # value of None for changes gives the series its style from before the popup
            current = viewer.values.subplots
            if row >= len(current) or position >= len(current[row]) or get_series_name(current[row][position]) != name:
                return
            subplots = list(current)
            if changes is None:
                # the item from before the popup, with its text, thus the step of Undo reverts to the same command
                subplots[row] = (*current[row][:position], item, *current[row][position + 1 :])
            else:
                newstyle = {flag.removeprefix("--"): value for flag, value in changes.items()}
                subplots[row] = set_item_style(current[row], position, newstyle)
            apply(dc.replace(viewer.values, subplots=tuple(subplots)), undoable=undoable)

        edit_series_properties(
            window,
            name,
            {f"--{key}": value for key, value in style.items()},
            swatch.defaultcolour if swatch is not None else "black",
            swatch.defaultlinewidth if swatch is not None else float(mpl.rcParams["lines.linewidth"]),
            show_changes,
            anchor=anchor,
            linewidthscale=get_linewidthscale(viewer.parser, viewer.values.otheroptions),
            # the label is a part of one plot item, thus a label that starts with "-" is no flag
            labelinlist=False,
        )

    def apply_subplots(subplots: "Sequence[tuple[str, ...]]") -> None:
        """Apply the subplots.

        A subplot with no item stays, and it draws an empty frame. --ionpoptype of the command waits for a populations
        subplot, and then it goes to each one.
        """
        newsubplots, otheroptions = move_poptype_to_subplots(
            tuple(subplots), viewer.values.otheroptions, viewer.modeldata.estimatorcolumns
        )
        apply(dc.replace(viewer.values, subplots=newsubplots, otheroptions=otheroptions))

    def set_subplot(row: int, subplot: tuple[str, ...]) -> None:
        subplots = list(viewer.values.subplots)
        subplots[row] = subplot
        apply_subplots(subplots)

    def is_shown_card(row: int, edit: QtWidgets.QLineEdit) -> bool:
        """Return True if a field is in the card that the screen shows for the row."""
        return row < len(cards) and cards[row].frame.isAncestorOf(edit)

    def on_subplot_type(row: int, seriestype: str) -> None:
        subplot = viewer.values.subplots[row]
        levelnames = get_levelnames(seriestype)
        set_subplot(row, change_subplot_type(subplot, seriestype, viewer.modeldata.estimatorcolumns, levelnames))

    def on_yrange(row: int, yminedit: QtWidgets.QLineEdit, ymaxedit: QtWidgets.QLineEdit) -> None:
        if not is_shown_card(row, yminedit):
            return
        if (texts := read_limit_fields([(yminedit, "y min"), (ymaxedit, "y max")], show_error)) is not None:
            subplot = viewer.values.subplots[row]
            set_subplot(row, replace_directives(subplot, {"ymin": texts[0] or None, "ymax": texts[1] or None}))

    def add_item(row: int, item: str) -> None:
        subplot = viewer.values.subplots[row]
        item = resolve_item_alias(item)
        directive = get_item_directive(item)
        if directive is not None:
            value = item.partition("=")[2].strip()
            if not value:
                show_error(f"Give a value after {item}, e.g. {directive}=1e-16")
                return
            set_subplot(row, replace_directives(subplot, {directive: value}))
        elif get_series_name(item) in get_subplot_names(subplot):
            show_error(f"The subplot already shows {get_series_name(item)}")
        else:
            items = get_series_items(subplot)
            # a name goes after the other names, thus the directives stay at the end
            set_subplot(
                row,
                (*items, item, *subplot[len(items) :]) if subplot[: len(items)] == tuple(items) else (*subplot, item),
            )

    def popup_has_pick(edit: QtWidgets.QLineEdit) -> bool:
        """Return True if the popup of the completer of a field shows a highlighted name.

        The Return key in the popup gives returnPressed first and activated second, thus only the handler of activated
        takes the name.
        """
        completer = edit.completer()
        popup = completer.popup() if completer is not None else None
        return popup is not None and popup.isVisible() and popup.currentIndex().isValid()

    def on_add_item(row: int, edit: QtWidgets.QLineEdit) -> None:
        if popup_has_pick(edit) or not is_shown_card(row, edit):
            return
        item = edit.text().strip()
        edit.clear()
        if item:
            add_item(row, item)

    def on_complete_item(row: int, edit: QtWidgets.QLineEdit, item: str) -> None:
        if not is_shown_card(row, edit):
            return
        edit.clear()
        add_item(row, item)

    def on_remove_item(row: int, position: int) -> None:
        apply_subplots(remove_subplot_item(viewer.values.subplots, row, position))

    def apply_subplot_order(order: "Sequence[int]") -> None:
        """Apply the subplots in a new order, which gives the old row of each subplot."""
        apply_subplots([viewer.values.subplots[oldrow] for oldrow in order])

    def on_delete_subplot(row: int) -> None:
        apply_subplot_order([oldrow for oldrow in range(len(viewer.values.subplots)) if oldrow != row])

    def move_subplot(row: int, newrow: int) -> None:
        order = list(range(len(viewer.values.subplots)))
        if row != newrow and 0 <= newrow < len(order):
            order.insert(newrow, order.pop(row))
            apply_subplot_order(order)

    def on_move_key(row: int, step: int) -> None:
        """Move the subplot one row up or down, and keep the focus on its header for the next key."""
        nonlocal pendingfocus
        if 0 <= row + step < len(viewer.values.subplots):
            pendingfocus = (row + step, "dragheader", None)
            move_subplot(row, row + step)

    def on_collapse_subplot(row: int) -> None:
        collapsedrows.symmetric_difference_update({row})
        show_subplots()

    def on_yaxis_toggle(row: int) -> None:
        openyrows.symmetric_difference_update({row})
        if row < len(cards):
            card = cards[row]
            isopen = row in openyrows
            set_disclosure_symbol(card.yaxisbutton, isopen=isopen)
            card.yaxisbox.setVisible(isopen)

    def on_reset_styles(row: int) -> None:
        set_subplot(row, remove_item_styles(viewer.values.subplots[row]))

    def get_chip_widget(row: int, position: int) -> QtWidgets.QWidget | None:
        if row >= len(cards):
            return None
        return next((chip for chipposition, chip in cards[row].chips if chipposition == position), None)

    def on_click_chip(row: int, position: int, *_: object) -> None:
        """Show the line properties of the series of a chip, under the chip."""
        chip = get_chip_widget(row, position)
        edit_item_style(row, position, chip.mapToGlobal(chip.rect().bottomLeft()) if chip is not None else None)

    def get_chip_drop_target(position: QtCore.QPoint) -> tuple[int, int] | None:
        """Return the row of the card and the index of the chip after a position on the screen, or None.

        An index equal to the number of chips gives the place after the last chip. A position outside the cards gives
        None.
        """
        for row, card in enumerate(cards):
            if not card.chipsbox.isVisible() or not card.frame.rect().contains(card.frame.mapFromGlobal(position)):
                continue
            point = card.chipsbox.mapFromGlobal(position)
            for index, (_, chip) in enumerate(card.chips):
                geometry = chip.geometry()
                if point.y() < geometry.top() or (point.y() <= geometry.bottom() and point.x() < geometry.center().x()):
                    return row, index
            return row, len(card.chips)
        return None

    def set_chip_fade(row: int, position: int, *, faded: bool) -> None:
        if (chip := get_chip_widget(row, position)) is not None:
            set_drag_fade(chip, faded=faded)

    def is_same_subplot_type(row: int, targetrow: int) -> bool:
        """Return True if a series of one subplot can move to the other subplot.

        A name takes the meaning of the type of its subplot, e.g. Te in a populations subplot is tellurium.
        """
        subplots, columns = viewer.values.subplots, viewer.modeldata.estimatorcolumns
        return get_subplot_seriestype(subplots[row], columns) == get_subplot_seriestype(subplots[targetrow], columns)

    def on_drag_chip(row: int, position: int, pointer: QtCore.QPoint) -> None:
        """Show a vertical line at the place where the dragged chip goes, if the chip can go there."""
        set_chip_fade(row, position, faded=True)
        target = get_chip_drop_target(pointer)
        if target is None or not is_same_subplot_type(row, target[0]):
            chipdropline.hide()
            return
        card = cards[target[0]]
        if target[1] < len(card.chips):
            chip = card.chips[target[1]][1]
            topleft = chip.mapTo(subplotsbox, QtCore.QPoint(-3, 0))
        elif card.chips:
            chip = card.chips[-1][1]
            topleft = chip.mapTo(subplotsbox, QtCore.QPoint(chip.width() + 1, 0))
        else:
            chip = card.chipsbox
            topleft = chip.mapTo(subplotsbox, QtCore.QPoint(0, 0))
        chipdropline.setGeometry(topleft.x(), topleft.y(), 2, max(chip.height(), 16))
        chipdropline.show()
        chipdropline.raise_()

    def on_drop_chip(row: int, position: int, pointer: QtCore.QPoint) -> None:
        chipdropline.hide()
        set_chip_fade(row, position, faded=False)
        if (target := get_chip_drop_target(pointer)) is not None:
            move_chip(row, position, *target)

    def move_chip(row: int, position: int, targetrow: int, chipindex: int) -> None:
        subplots, columns = viewer.values.subplots, viewer.modeldata.estimatorcolumns
        if not is_same_subplot_type(row, targetrow):
            show_error("A series can move only to a subplot of the same type")
            return
        moved = move_series_item(subplots, (row, position), targetrow, chipindex, columns)
        if moved is None:
            show_error(f"The subplot already shows {get_series_name(subplots[row][position])}")
        elif moved != subplots:
            apply_subplots(moved)

    def on_move_chip(row: int, position: int, step: int) -> None:
        """Swap the series of a chip and of the adjacent chip, for Alt-Up and Alt-Down.

        The viewer makes the card again. The moved chip then takes the focus, thus the next key moves the same series.
        """
        nonlocal pendingfocus
        subplot = viewer.values.subplots[row]
        chippositions = [chipposition for chipposition, _ in get_chip_items(subplot, viewer.modeldata.estimatorcolumns)]
        if position not in chippositions or not 0 <= (index := chippositions.index(position) + step) < len(
            chippositions
        ):
            return
        other = chippositions[index]
        items = list(subplot)
        items[position], items[other] = items[other], items[position]
        pendingfocus = (row, "chip", other)
        set_subplot(row, tuple(items))

    def get_focus_row() -> int:
        """Return the row of the card with the keyboard focus, or else of the last card."""
        focuswidget = QtWidgets.QApplication.focusWidget()
        return next(
            (row for row, card in enumerate(cards) if focuswidget is not None and card.frame.isAncestorOf(focuswidget)),
            len(cards) - 1,
        )

    def on_add_series_key() -> None:
        """Give the keyboard to the field that adds a series to the card with the focus, or else to the last card."""
        nonlocal pendingfocus
        row = get_focus_row()
        if row in collapsedrows:
            # the card that opens is a new card, and show_subplots gives the focus after this function
            pendingfocus = (row, "add", None)
            on_collapse_subplot(row)
        if (addedit := cards[row].frame.findChild(QtWidgets.QLineEdit, "add")) is None:
            return
        scrollarea = addedit.parentWidget()
        while scrollarea is not None and not isinstance(scrollarea, QtWidgets.QScrollArea):
            scrollarea = scrollarea.parentWidget()
        if scrollarea is not None:
            scrollarea.ensureWidgetVisible(addedit)
        addedit.setFocus()

    def on_add_subplot_key() -> None:
        """Add a subplot below the card with the keyboard focus, or else after the last card."""
        insert_new_subplot(get_focus_row() + 1)

    def get_drop_index(position: QtCore.QPoint) -> int:
        """Return the index of the gap between the cards that is nearest to a position on the screen."""
        y = subplotsbox.mapFromGlobal(position).y()
        return sum(card.frame.geometry().center().y() < y for card in cards)

    def on_drag_subplot(row: int, position: QtCore.QPoint) -> None:
        index = get_drop_index(position)
        spacing = subplotslayout.spacing()
        if index < len(cards):
            y = cards[index].frame.geometry().top() - (spacing + 2) // 2
        else:
            y = cards[-1].frame.geometry().bottom() + (spacing + 2) // 2
        dropline.setGeometry(0, min(max(y, 0), subplotsbox.height() - 2), subplotsbox.width(), 2)
        dropline.show()
        dropline.raise_()
        # a disabled header gets no mouse events, thus the dragged card fades but stays enabled
        set_drag_fade(cards[row].frame, faded=True)

    def on_drop_subplot(row: int, position: QtCore.QPoint) -> None:
        dropline.hide()
        set_drag_fade(cards[row].frame, faded=False)
        index = get_drop_index(position)
        move_subplot(row, index if index <= row else index - 1)

    def on_directive_selector(row: int, directive: str, defaulttext: str, text: str) -> None:
        subplot = viewer.values.subplots[row]
        set_subplot(row, replace_directives(subplot, {directive: None if text == defaulttext else text}))

    def insert_subplot(row: int, subplot: tuple[str, ...]) -> None:
        """Insert a subplot at row, and give the keyboard to the field of its card that adds a series."""
        nonlocal pendingfocus
        pendingfocus = (row, "add", None)
        subplots = viewer.values.subplots
        apply_subplots([*subplots[:row], subplot, *subplots[row:]])

    def insert_new_subplot(row: int) -> None:
        insert_subplot(
            row,
            get_new_subplot(
                viewer.values.subplots, viewer.modeldata.defaultsubplots, viewer.modeldata.estimatorcolumns
            ),
        )

    def add_suggested_subplot(subplot: tuple[str, ...]) -> None:
        insert_subplot(len(viewer.values.subplots), subplot)

    def set_directives(row: int, directives: "Mapping[str, str | None]") -> None:
        subplots = list(viewer.values.subplots)
        subplots[row] = replace_directives(subplots[row], directives)
        apply(dc.replace(viewer.values, subplots=tuple(subplots)))

    def on_set_current_range(row: int) -> None:
        """Set ymin= and ymax= of a subplot to the range of its colour scale or of its y axis on the screen."""
        limits: list[tuple[float | None, float | None]]
        if viewer.isimage:
            label = get_panel_axes_label(row)
            limits = [
                mesh.get_clim()
                for axis in viewer.fig.axes
                if axis.get_label() == label
                for mesh in axis.collections
                if isinstance(mesh, QuadMesh)
            ]
        else:
            frames = get_plot_frames(viewer.fig)
            limits = [frames[row].get_ylim()] if row < len(frames) else []
        # a panel with no value has no colour scale
        finitelimits = [
            (float(low), float(high))
            for low, high in limits
            if low is not None and high is not None and np.isfinite(low) and np.isfinite(high)
        ]
        if not (plot_shows_values() and finitelimits):
            show_error("This subplot is not on the screen yet. Wait for the plot, then try again")
            return
        set_directives(
            row,
            {
                "ymin": format(min(low for low, _ in finitelimits), ".6g"),
                "ymax": format(max(high for _, high in finitelimits), ".6g"),
            },
        )

    def show_subplot_section() -> None:
        """Show a card for each subplot, the suggestions of a new subplot, and the state of the default subplots."""
        values = viewer.values
        show_subplots()
        defaultbutton.setEnabled(
            values.subplots != viewer.modeldata.defaultsubplots and bool(viewer.modeldata.defaultsubplots)
        )
        skippedcount = len(viewer.modeldata.skippeddefaults)
        set_note_text(
            skippeddefaultslabel,
            f"The default subplots leave out {skippedcount} {'plot' if skippedcount == 1 else 'plots'}."
            if skippedcount
            else "",
            "\n".join(["The default subplots leave out:", *(f"• {note}" for note in viewer.modeldata.skippeddefaults)]),
        )

    defaultbutton.clicked.connect(lambda: apply_subplots(viewer.modeldata.defaultsubplots))
    return show_subplot_section, edit_item_style, set_directives, on_add_series_key, on_add_subplot_key
