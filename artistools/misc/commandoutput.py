"""Give the product of a command: a table on the standard output, a data file, or a set of numbered frames."""

import argparse
import dataclasses as dc
import os
import sys
import typing as t
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType

from artistools.misc.columns import get_column_description_lines
from artistools.misc.console import confirm_overwrite_of_files
from artistools.misc.console import print_warning
from artistools.misc.fileio import combine_frames
from artistools.misc.fileio import get_file_copies
from artistools.misc.fileio import print_saved
from artistools.misc.fileio import write_parquet_atomic
from artistools.misc.fileio import write_text_file_atomic

if t.TYPE_CHECKING:
    import polars as pl


def make_output_folder(folder: Path | str, verb: str) -> Path:
    """Make the folder that holds the output of a command, and refuse a name that a file holds."""
    folder = Path(folder)
    if folder.exists() and not folder.is_dir():
        msg = f"'{folder}' names a file that exists, and this command {verb} a folder of that name"
        raise ValueError(msg)

    folder.mkdir(parents=True, exist_ok=True)

    return folder


def confirm_overwrite(folders: Iterable[Path | str], filenames: Sequence[str], force: bool) -> None:
    """Warn about the output files that exist, and stop the command if the user does not agree to overwrite them.

    The output files are the file names in each of the folders. Each name also stands for its compressed copies,
    e.g. model.txt for model.txt.zst. A command calls this function before the work starts, thus a refusal costs no
    time. Nobody can answer when stdin is not a terminal, thus the command then stops if it has no --force.
    """
    confirm_overwrite_of_files(
        [
            path
            for folder in folders
            for filename in filenames
            for path in get_file_copies(Path(folder, filename))
            if path.is_file()
        ],
        force,
    )


def confirm_overwrite_matches(folder: Path | str, patterns: Sequence[str], force: bool) -> None:
    """Apply confirm_overwrite to the files of folder that match the glob patterns, e.g. "spectrum_ts*.txt".

    A command whose file names hold a timestep or a value from the data takes patterns, because the command does not
    know the names before the work starts.
    """
    confirm_overwrite_of_files(
        sorted({path for pattern in patterns for path in Path(folder).glob(pattern) if path.is_file()}), force
    )


def print_product(args: argparse.Namespace, *values: object) -> None:
    """Print the product of a command, which --quiet keeps.

    --quiet sends the progress messages to the null device. The product of the command, e.g. the table
    of --print-data or the listing of --listvariables, must reach the standard output even so. A script
    then reads that product with no progress message around it.
    """
    productstream = getattr(args, "productstream", None) or sys.stdout
    if len(values) == 1 and not isinstance(values[0], str):
        import polars as pl

        if isinstance(values[0], pl.DataFrame):
            write_data_table(values[0], productstream, outputformat=getattr(args, "output_format", None))
            return

    print(*values, file=productstream)


def get_json_rows(dfdata: "pl.DataFrame") -> list[dict[str, t.Any]]:
    """Return the rows of a table for JSON, with null for a missing value and for an infinite value.

    JSON has no NaN and no infinity. json.dumps wrote Infinity, which a strict parser of JSON refuses.
    """
    import polars as pl

    floatcolumns = [name for name, dtype in dfdata.schema.items() if dtype.is_float()]
    return dfdata.with_columns(
        pl.when(pl.col(name).is_finite()).then(pl.col(name)).alias(name) for name in floatcolumns
    ).to_dicts()


# the formats of --output-format. "text" has a space between the values, and "csv" has a comma
OUTPUT_FORMATS: t.Final = ("text", "json", "csv")


def write_data_table(
    dfdata: "pl.DataFrame",
    destination: "Path | t.TextIO",
    commentlines: Sequence[str] = (),
    outputformat: str | None = None,
) -> None:
    """Write a table as text with a space between the values and a header row of column names.

    The polars display of a table shows only some rows, thus a reader of the output does not get each value. A reader
    that splits on spaces moves each later value to the wrong column after an empty field. Thus a missing value is nan.
    A value or a column name that contains a space has quotes, which polars and pandas read. Each line of commentlines
    goes before the table with "# " at its start, which pl.read_csv(comment_prefix="#") and np.loadtxt skip. A file
    replaces a symbolic link at its path, because write_text_file_atomic does not write through a link.

    A file with the suffix .json, .csv, or .parquet takes that format, as the suffix of a figure sets its format.
    outputformat, e.g. from --output-format, replaces the format of the suffix. A JSON file holds an object with the
    keys "comments" and "rows", and a missing value is null. A parquet file keeps commentlines in its metadata key
    "comments".
    """
    suffix = destination.suffix.lower() if isinstance(destination, Path) else ""
    if isinstance(destination, Path) and suffix == ".parquet" and outputformat is None:
        write_parquet_atomic(dfdata, destination, metadata={"comments": "\n".join(commentlines)})
        return
    tableformat = outputformat or {".json": "json", ".csv": "csv"}.get(suffix, "text")

    def write_to_stream(stream: t.TextIO) -> None:
        if tableformat == "json":
            import json

            json.dump({"comments": list(commentlines), "rows": get_json_rows(dfdata)}, stream, default=str, indent=1)
            stream.write("\n")
            return
        stream.writelines(f"# {line}\n" for line in commentlines)
        if tableformat == "csv":
            dfdata.write_csv(stream)
            return
        dfdata.write_csv(stream, separator=" ", null_value="nan")

    if isinstance(destination, Path):
        write_text_file_atomic(destination, write_to_stream)
    else:
        write_to_stream(destination)


def resolve_table_outputfile(args: argparse.Namespace, defaultname: str) -> Path | None:
    """Return the file that -o names, or None when the user gives no -o. Without -o, the command prints its table.

    A -o path with no file extension names a folder, which takes defaultname. The function does the overwrite check
    here, before the work starts. -f skips the check.
    """
    if not args.outputfile:
        return None
    outputfile = resolve_outputfile(args.outputfile, defaultname)
    confirm_overwrite([outputfile.parent], [outputfile.name], force=getattr(args, "force", False))
    return outputfile


def print_or_write_table(
    args: argparse.Namespace, dftable: "pl.DataFrame", outputfile: Path | None, commentlines: Sequence[str] = ()
) -> None:
    """Print the table with no -o, or write it to outputfile with the provenance lines and commentlines.

    A printed JSON table holds commentlines under its key "comments", as a JSON file does. A parser of JSON then reads
    the whole output, and a line of text in front of the object would stop it.
    """
    if outputfile is None:
        if getattr(args, "output_format", None) == "json":
            productstream = getattr(args, "productstream", None) or sys.stdout
            write_data_table(dftable, productstream, commentlines=commentlines, outputformat="json")
            return
        print_product(args, dftable)
        return

    make_output_folder(outputfile.parent, "writes")
    write_data_table(
        dftable,
        outputfile,
        commentlines=[*get_provenance_lines(args), *commentlines],
        outputformat=getattr(args, "output_format", None),
    )
    print_saved(outputfile)


def get_provenance_lines(args: argparse.Namespace) -> list[str]:
    """Return the lines that tell how a command made its data: the version, the time, the folder, and the command.

    A relative model path in the command starts at the working folder, thus a line gives that folder too. A call from
    Python has no command line, thus the line then gives each argument that holds a plain value.
    """
    import datetime as dt
    from importlib.metadata import version

    timetext = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    commandline = getattr(args, "commandline", None)
    if commandline is None:
        plainvalues = (str, int, float, bool, Path, list, tuple, type(None))
        argtexts = [f"{key}={value!r}" for key, value in sorted(vars(args).items()) if isinstance(value, plainvalues)]
        commandline = f"a Python call with the arguments {', '.join(argtexts)}"
    return [
        f"artistools {version('artistools')}, {timetext}",
        f"working folder: {Path.cwd()}",
        f"command: {commandline}",
    ]


def output_data(args: argparse.Namespace, tables: "Mapping[str, pl.DataFrame]", outputfile: Path | str) -> None:
    """Print the plotted data, write it to files, or both, as --print-data and --write-data ask.

    Each key of tables is the end of a file name, e.g. "_residuals" gives myplot_residuals.txt for myplot.pdf, and an
    empty key gives myplot.txt. The standard output gives the tables in the same order, with an empty line between two
    tables. --output-format json prints one JSON object, with the rows of each table under its key without the "_",
    e.g. "residuals", and "data" for the empty key. A parser of JSON reads one document alone. --output-format csv
    prints the main table alone, with a warning that names the other tables. A reader of CSV takes one header.

    --output-format also gives the format and the suffix of each file, e.g. myplot.json. The function skips an empty
    table. A file starts with comment lines that tell how the command made the data and what each column holds. A pipe
    reads the standard output, thus it gets no comment lines. An empty table replaces the file of an earlier run, thus
    that file cannot seem to hold the data of this run.
    """
    outputformat = getattr(args, "output_format", None)
    printedtables = {suffix: dfdata for suffix, dfdata in tables.items() if not dfdata.is_empty()}
    # a command can give one figure for each element or each time, thus run_command prints the tables of all the
    # figures as one document when the command ends
    if getattr(args, "print_data", False) and (collected := getattr(args, "collectedtables", None)) is not None:
        collected.append((Path(outputfile).name, printedtables))
    elif getattr(args, "print_data", False) and outputformat == "json":
        import json

        document = {suffix.lstrip("_") or "data": get_json_rows(dfdata) for suffix, dfdata in printedtables.items()}
        print_product(args, json.dumps(document, default=str, indent=1))
    elif getattr(args, "print_data", False) and outputformat == "csv" and len(printedtables) > 1:
        # a reader of CSV takes one header, thus the output holds the main table alone
        mainsuffix = "" if "" in printedtables else next(iter(printedtables))
        othernames = ", ".join(suffix.lstrip("_") for suffix in printedtables if suffix != mainsuffix)
        print_product(args, printedtables[mainsuffix])
        print_warning(
            f"--output-format csv prints the main table alone, and the other tables stay out: {othernames}."
            " --output-format json prints each table, and --write-data writes one file for each table"
        )
    elif getattr(args, "print_data", False):
        for tableindex, dfdata in enumerate(printedtables.values()):
            if tableindex > 0:
                print_product(args, "")
            print_product(args, dfdata)
    if getattr(args, "write_data", False):
        filesuffix = {"json": ".json", "csv": ".csv"}.get(outputformat or "text", ".txt")
        for suffix, dfdata in tables.items():
            datafile = Path(outputfile).with_name(f"{Path(outputfile).stem}{suffix}{filesuffix}")
            if dfdata.is_empty() and not os.path.lexists(datafile):
                continue
            emptylines = (
                ["the table has no rows, e.g. because the plot has no point in the selected range"]
                if dfdata.is_empty()
                else []
            )
            commentlines = [*get_provenance_lines(args), *emptylines, *get_column_description_lines(dfdata.columns)]
            write_data_table(dfdata, datafile, commentlines=commentlines, outputformat=outputformat)
            print_saved(datafile)


def print_collected_data(args: argparse.Namespace) -> None:
    """Print the tables that output_data collected for each figure as one document of --output-format json or csv.

    A reader of JSON or of CSV takes one document, thus the rows of the tables with one key join one table. With more
    than one figure, the column "figure" gives the file name of the figure of each row. A table can hold no column of
    the timestep or of the element, e.g. a table of plotnltepops, and the file name gives them.
    """
    import polars as pl

    collected: list[tuple[str, Mapping[str, pl.DataFrame]]] = getattr(args, "collectedtables", None) or []
    framesofkey: dict[str, list[pl.DataFrame]] = {}
    for figurename, tables in collected:
        for suffix, dfdata in tables.items():
            framesofkey.setdefault(suffix, []).append(
                dfdata.with_columns(figure=pl.lit(figurename)) if len(collected) > 1 else dfdata
            )
    tables = {suffix: pl.concat(frames, how="diagonal_relaxed") for suffix, frames in framesofkey.items()}
    printargs = argparse.Namespace(**(vars(args) | {"write_data": False, "collectedtables": None}))
    output_data(printargs, tables, "")


def resolve_outputfile(outputfile: Path | str | None, defaultoutputfile: Path | str) -> Path:
    """Return the output file path, appending the default filename if outputfile is unset or refers to a folder.

    A path with no file extension is treated as a folder and will be created if it does not exist.
    """
    if not outputfile:
        return Path(defaultoutputfile)

    outputfile = Path(outputfile)
    if outputfile.is_dir() or not outputfile.suffixes:
        return make_output_folder(outputfile, "needs") / defaultoutputfile

    return outputfile


def get_template_fields(template: Path | str) -> list[str]:
    """Return the names of the fields that a template of an output name holds."""
    import re

    return re.findall(r"\{(\w+)", str(template))


# the older name of each output template field, which a script can still hold. The help names one
# spelling for each field, thus the message that lists them leaves these out
OLDTEMPLATEFIELDS: t.Final[Mapping[str, str]] = MappingProxyType({"modelgridindex": "cell", "time_days": "timedays"})


def format_frame_path(frametemplate: Path | str, **fields: t.Any) -> str:
    """Return the path of one frame, and name the fields of the command for a template that holds another.

    A user writes the template, thus a field that the command does not give is a mistake of the
    arguments and not a fault of artistools.
    """
    fieldsandoldnames = fields | {old: fields[new] for old, new in OLDTEMPLATEFIELDS.items() if new in fields}
    try:
        return str(frametemplate).format(**fieldsandoldnames)
    except KeyError as exc:
        given = ", ".join(f"{{{name}}}" for name in fields)
        msg = f"The name of the output holds the field {exc}, and this command gives {given}"
        raise ValueError(msg) from exc
    except IndexError as exc:
        given = ", ".join(f"{{{name}}}" for name in fields)
        msg = f"The name of the output holds a field with no name, and each field needs one of: {given}"
        raise ValueError(msg) from exc


@dc.dataclass(frozen=True, slots=True)
class FrameSet:
    """The frames of a run that draws several figures, and the product that holds them.

    One value says whether the run combines its frames. The command reads it for the figure of each
    frame, and finish makes the product, thus the three parts of the run cannot disagree.
    """

    frametemplate: Path
    productpath: Path | None
    combines: bool
    gifduration: float | None = None

    def finish(self, framepaths: "Sequence[str | Path]", args: argparse.Namespace) -> Path | str | None:
        """Combine the frames into the product, for a run that combines them, and give its path."""
        if not self.combines:
            return None

        return combine_frames(
            framepaths, self.productpath, openfile=getattr(args, "open", False), gifduration=self.gifduration
        )


def resolve_frameset_paths(
    outputfile: Path | str | None,
    *,
    framecount: int,
    framename: str,
    productname: str | None = None,
    combines: bool = False,
    gifduration: float | None = None,
) -> FrameSet:
    """Return the frames of the run and the product that holds them.

    A run that draws several figures combines them into one product, e.g. a gif or a merged pdf.
    combines says that such a product comes, and productname gives it a name for a -o path that names a
    folder. Without that name the combining step names it, as merge_pdf_files takes the names of the
    first frame and the last one.

    For a run that combines its frames, a -o path that has a file extension names the product itself, thus the
    frames go in the folder that holds it. A -o path with no file extension names a folder. This makes that folder
    either way. A -o name that holds a field, e.g. frame_{timestep}.pdf, names each frame and never the product.
    A run that does not combine its frames makes no product, thus its -o path names the frame.
    """
    givenpath = Path(outputfile) if outputfile else Path()

    if combines and givenpath.suffix and not givenpath.is_dir() and "{" not in givenpath.name:
        # a merge of the frames always writes pdf data, thus a different suffix would name the format incorrectly
        if gifduration is None and givenpath.suffix.lower() != ".pdf":
            msg = (
                f"'{givenpath.name}' names the merged product of {framecount} frames, and a merge writes a pdf file."
                f" Give a name that ends with .pdf, or a folder with -o"
            )
            raise ValueError(msg)

        # the folder of the product can carry a suffix of its own, e.g. results.v1, thus make it here
        # and let resolve_outputfile read it as a folder and not as the name of one frame
        givenpath.parent.mkdir(parents=True, exist_ok=True)

        return FrameSet(resolve_outputfile(givenpath.parent, framename), givenpath, combines, gifduration)

    frametemplate = resolve_outputfile(outputfile, framename)
    if framecount > 1 and "{" not in frametemplate.name:
        fields = get_template_fields(framename)
        example = f"-o 'frame_{{{fields[0]}}}{Path(framename).suffix}'" if fields else "-o myfolder"
        msg = (
            f"'{frametemplate.name}' names one file, and this command writes {framecount} frames. Give "
            f"a folder with -o, or a name that holds a field, e.g. {example}"
        )
        raise ValueError(msg)

    productpath = frametemplate.parent / productname if productname is not None else None

    return FrameSet(frametemplate, productpath, combines, gifduration)
