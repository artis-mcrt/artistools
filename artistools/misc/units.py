"""Read a value with a unit from the command line, e.g. a time of 300d or a wavelength of 600nm."""

import argparse
import math
import re
import typing as t
from collections.abc import Mapping
from types import MappingProxyType

from artistools.constants import C_cm_per_s
from artistools.constants import day_to_s
from artistools.constants import EV_to_erg
from artistools.constants import hour_to_s
from artistools.constants import km_to_cm

# a number, then an optional unit, e.g. 600nm, 2.5 d, or 1e5. The unit starts with a character that no number holds
VALUE_WITH_UNIT_PATTERN: t.Final = re.compile(r"\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([^\d\s.+-].*?)?\s*")


# the size of each unit of a time value in days. A value with no unit has the unit "", which is days
TIME_UNITS_IN_DAYS: t.Final[Mapping[str, float]] = MappingProxyType({
    "": 1.0,
    "d": 1.0,
    "day": 1.0,
    "days": 1.0,
    "h": hour_to_s / day_to_s,
    "hour": hour_to_s / day_to_s,
    "hours": hour_to_s / day_to_s,
    "s": 1.0 / day_to_s,
})


# the help text of each option that takes a time
TIME_UNIT_HELP: t.Final = "A value with no unit is in days. A unit is optional, e.g. 12h"


# the help text of each option that takes a wavelength in Angstroms
WAVELENGTH_UNIT_HELP: t.Final = "A value with no unit is in Angstroms. A unit is optional, e.g. 600nm"


# a hyphen that ends a time and starts the next one. A hyphen of an exponent, e.g. 1e-2, follows a digit and an e
TIME_RANGE_SEPARATOR: t.Final = re.compile(r"(?<=[0-9.a-zA-Z])(?<![0-9.][eE])-")


def split_value_and_unit(text: str) -> tuple[float, str]:
    """Return the number and the unit of a command line value, e.g. (600.0, "nm") for 600nm.

    A value with no unit gives the unit "". A text that does not start with a number gives a ValueError.
    """
    if (match := VALUE_WITH_UNIT_PATTERN.fullmatch(text)) is None:
        msg = f"{text!r} is not a number with an optional unit"
        raise ValueError(msg)
    return float(match.group(1)), match.group(2) or ""


def time_days_arg(text: str) -> float:
    """Return the time in days of a command line value, e.g. 300, 300d, 12h, or 2.6e7s.

    A value with no unit is in days.
    """
    unitnames = ", ".join(name for name in TIME_UNITS_IN_DAYS if len(name) == 1)
    try:
        number, unit = split_value_and_unit(text)
    except ValueError:
        msg = f"'{text}' is not a time. Give a number of days, or a number with one of the units {unitnames}"
        raise argparse.ArgumentTypeError(msg) from None
    if unit.lower() not in TIME_UNITS_IN_DAYS:
        msg = f"'{unit}' in '{text}' is not a unit of time. Give one of the units {unitnames}, or no unit for days"
        raise argparse.ArgumentTypeError(msg)
    return number * TIME_UNITS_IN_DAYS[unit.lower()]


def time_days_range_arg(text: str) -> str:
    """Return a time or a time range in days, e.g. "250-300" for 250d-300d. A text with no unit stays the same.

    A command reads the time text after the parse, e.g. with parse_timedays_range. Thus the text with a unit becomes
    the same text in days, and a text with no unit keeps its form, e.g. in the label of a plot. deposition also reads
    a list such as 1,5,30, thus each item of a list takes the same conversion.
    """
    if "," in text:
        return ",".join(time_days_range_arg(item) for item in text.split(","))
    try:
        float(text)
    except ValueError:
        pass
    else:
        return text
    parts = TIME_RANGE_SEPARATOR.split(text.strip())
    if len(parts) > 2:
        msg = f"'{text}' is not a time or a time range. Give a time or a time range, e.g. 300, 300d, or 250d-300d"
        raise argparse.ArgumentTypeError(msg)
    days = [time_days_arg(part) for part in parts]
    if not any(split_value_and_unit(part)[1] for part in parts):
        return text
    return "-".join(f"{value:.10g}" for value in days)


def wavelength_angstroms_arg(text: str) -> float:
    """Return the wavelength in Angstroms of a command line value, e.g. 6000, 6000A, 600nm, or 0.6micron.

    A value with no unit is in Angstroms. A unit is one of the wavelength units of --xunit of plotspectra.
    """
    unitnames = ", ".join(name for name, xunit in XUNITS.items() if xunit.kind == "wavelength")
    try:
        number, unit = split_value_and_unit(text)
        xunit = XUNITS[convert_xunit_aliases_to_canonical(unit)] if unit else XUNITS["angstrom"]
    except ValueError:
        msg = f"'{text}' is not a wavelength. Give a number of Angstroms, or a number with one of the units {unitnames}"
        raise argparse.ArgumentTypeError(msg) from None
    if xunit.kind != "wavelength":
        msg = f"'{text}' is not a wavelength. Give one of the units {unitnames}, or no unit for Angstroms"
        raise argparse.ArgumentTypeError(msg)
    return number * xunit.unit_in_base_units


class XUnit(t.NamedTuple):
    """One unit of the horizontal axis of a spectrum, and the spellings that a user can give for it.

    The conversions work in a base unit of each kind: the angstrom for a wavelength, the hertz for
    a frequency, and the electronvolt for an energy. unit_in_base_units gives the size of this unit
    in that base unit, e.g. 10.0 for a nanometre and 1000.0 for a kiloelectronvolt.
    """

    kind: t.Literal["wavelength", "frequency", "energy"]
    unit_in_base_units: float
    label: str
    aliases: tuple[str, ...]


# one table gives the conversion factor and the axis label. The unit message and the name
# suggestion also read it, thus a new unit needs one entry only
XUNITS: t.Final[Mapping[str, XUnit]] = MappingProxyType({
    # each canonical name is singular, as micron and erg are. A script can still give angstroms
    "angstrom": XUnit("wavelength", 1.0, "\u00c5", ("angstroms", "a", "ang", "\u00e5", "\u00e5ngstr\u00f6m")),
    "nm": XUnit("wavelength", 10.0, "nm", ("nanometer", "nanometers")),
    "micron": XUnit("wavelength", 10000.0, "\u03bcm", ("microns", "mu", "\u03bc", "\u03bcm")),
    "hz": XUnit("frequency", 1.0, "Hz", ()),
    "erg": XUnit("energy", 1.0 / EV_to_erg, "erg", ("ergs",)),
    "ev": XUnit("energy", 1.0, "eV", ("electronvolt",)),
    "kev": XUnit("energy", 1.0e3, "keV", ("kiloelectronvolt",)),
    "mev": XUnit("energy", 1.0e6, "MeV", ("megaelectronvolt",)),
})


def get_xunit_names() -> list[str]:
    """Return every spelling of a unit of the horizontal axis that a user can give."""
    return [name for canonical, unit in XUNITS.items() for name in (canonical, *unit.aliases)]


def get_xunit(xunit: str) -> XUnit:
    """Return the unit of the horizontal axis for its canonical name or an alias, e.g. "angstroms" or "nanometers"."""
    return XUNITS[convert_xunit_aliases_to_canonical(xunit)]


def parse_xunit_argument(value: str) -> str:
    """Return the canonical unit of the horizontal axis, or refuse a name that no unit takes.

    argparse calls this while it reads the command line, thus a name with a mistake stops the
    command before it reads a file. convert_xunit_aliases_to_canonical guards the keyword of the API.
    """
    from artistools.misc.console import suggest_names

    try:
        return convert_xunit_aliases_to_canonical(value)
    except ValueError:
        suggestion = suggest_names(value, get_xunit_names()) or f"The units are {', '.join(XUNITS)}"
        msg = f"'{value}' is not a unit of the horizontal axis. {suggestion}"
        raise argparse.ArgumentTypeError(msg) from None


def convert_xunit_aliases_to_canonical(xunit: str) -> str:
    """Return the canonical spelling of a spectrum x-axis unit name."""
    lowered = xunit.lower()
    for canonical, unit in XUNITS.items():
        if lowered == canonical or lowered in unit.aliases:
            return canonical

    msg = f"Unknown xunit {xunit}"
    raise ValueError(msg)


# -velocityunit auto takes c for a model with a vmax of this fraction of c or more, e.g. a kilonova
AUTO_VELOCITYUNIT_C_MIN_ON_C: t.Final = 0.2


def get_auto_velocity_unit(vmax_cmps: float) -> t.Literal["kmps", "c"]:
    """Return c for a model with a vmax of AUTO_VELOCITYUNIT_C_MIN_ON_C c or more, e.g. a kilonova, else km/s."""
    return "c" if vmax_cmps / C_cm_per_s >= AUTO_VELOCITYUNIT_C_MIN_ON_C else "kmps"


def parse_velocity_argument(value: str, *, requireunit: bool = False) -> tuple[float, t.Literal["kmps", "c"]]:
    """Return the velocity [km/s] and the unit of a command line value, e.g. 5000km/s or 0.1c.

    A number with no unit is in km/s. With requireunit, a number with no unit gives an error.
    """
    text = value.strip()
    unit: t.Literal["kmps", "c"]
    if text.lower().endswith("km/s"):
        numbertext, unit = text[: -len("km/s")], "kmps"
    elif text.lower().endswith("c"):
        numbertext, unit = text[:-1], "c"
    else:
        numbertext, unit = text, "kmps"
    helptext = "Give a number that ends in km/s, e.g. 5000km/s, or a fraction of c, e.g. 0.1c"
    if requireunit and numbertext == text:
        msg = f"'{value}' has no unit. {helptext}"
        raise argparse.ArgumentTypeError(msg)
    try:
        number = float(numbertext)
    except ValueError:
        number = math.nan
    if not math.isfinite(number):
        msg = f"'{value}' is not a finite velocity. {helptext}"
        raise argparse.ArgumentTypeError(msg)

    return (number * C_cm_per_s / km_to_cm if unit == "c" else number), unit


def get_velocity_label(velocity_kmps: float, unit: t.Literal["kmps", "c"]) -> str:
    """Return a velocity in the unit that the user gave, e.g. 0.1c or 5000 km/s."""
    return f"{velocity_kmps * km_to_cm / C_cm_per_s:g}c" if unit == "c" else f"{velocity_kmps:g} km/s"


def positive_velocity_kmps_arg(text: str) -> float:
    """Return the velocity [km/s] of a command line value with its unit, e.g. 0.1c or 30000km/s, when it is above 0.

    A bare number gives an error, because its unit is not clear. The unit of the axis can come from the model, with
    --velocityunit auto.
    """
    velocity_kmps, _ = parse_velocity_argument(text, requireunit=True)
    if velocity_kmps <= 0.0:
        msg = f"'{text}' is not a velocity above 0"
        raise argparse.ArgumentTypeError(msg)
    return velocity_kmps


def velocity_kmps_arg(text: str) -> float:
    """Return the velocity [km/s] of a command line value, e.g. 5000, 5000km/s, or 0.02c. A bare number is in km/s."""
    return parse_velocity_argument(text)[0]
