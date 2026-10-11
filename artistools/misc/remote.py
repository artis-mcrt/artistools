# ruff:file-ignore[suspicious-pickle-import, suspicious-pickle-usage]
"""Run the readers of a model on the host that holds the model.

A model path of the form "host:path" names a folder on a host that ssh reaches, e.g.
"vae26:~/scratch/mymodel". artistools starts "artistools server" on that host through ssh, and a
reader with the decorator on_model_host runs on that server. Only the result of the reader comes back,
thus the command copies no model file. The parquet caches stay beside the data on the remote host.

The server unpickles only the requests of the ssh process of the user. The client unpickles the replies with a
restricted unpickler, because a different user can control a host that the client reaches.
"""

import argparse
import functools
import os
import pickle
import re
import threading
import typing as t
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path

if t.TYPE_CHECKING:
    import subprocess

    import polars as pl


def make_remotepath_pattern(*, windows: bool) -> re.Pattern[str]:
    r"""Return the pattern of a remote path, with the rule of rsync.

    A colon before the first slash makes a remote path, e.g. "vae26:~/mymodel" and "user@vae26:/lustre/mymodel". A
    local path with such a colon starts with "./", e.g. "./run:2". An IPv6 address is in brackets, e.g.
    "user@[2001:db8::1]:/lustre/mymodel". On Windows, a host of one letter in front of a path separator is a drive,
    e.g. "C:\Users\me\mymodel". On a different system, it is an ssh host alias, e.g. "a:/lustre/mymodel".

    An ssh host, a user name, and an ssh host alias hold no space and no "=". Thus a label such as "Model B: Fe/Ni" is
    no remote path, and a series style of plotestimators such as "Te@color=tab:red" is no remote path.
    """
    drivelookahead = r"(?![A-Za-z]:[\\/])" if windows else ""
    return re.compile(
        rf"^{drivelookahead}(?P<host>(?:[^/:@=\[\s]*@)?\[[^\]/\s]*\]|[^/:=\[\s]+):(?P<path>.*)$", re.DOTALL
    )


REMOTEPATH_PATTERN = make_remotepath_pattern(windows=os.name == "nt")

# a user can give a different command to start the server, e.g. the path of an artistools in a clone
SERVER_COMMAND_ENVVAR = "ARTISTOOLS_REMOTE_COMMAND"

# the repository that a host can get a commit of a clone from
REPOSITORY_URL = "https://github.com/artis-mcrt/artistools"

# the servers of this process, with the lock of the pipes of each one. The key holds the process ID, because a
# child process cannot use the pipes of its parent. A server that fails leaves the registry, and the servers of the
# other hosts stay. An ssh start takes about 10 s, thus the process keeps one server for each host
SERVERS: "dict[tuple[str, int], tuple[subprocess.Popen[bytes], threading.Lock]]" = {}
# two threads can make the first call to one host at the same time. The lock makes the start of its server happen
# once
SERVERS_LOCK = threading.Lock()

# the modules of the objects that a reply of the server can hold. The unpickler of the client imports no module, and
# it reads only a module that this process has loaded already
REPLY_MODULES = frozenset({
    "argparse",
    "artistools",
    "builtins",
    "collections",
    "copyreg",
    "datetime",
    "numpy",
    "pathlib",
})

# the modules of the exceptions that a reply can hold. The server sends a different exception as a RuntimeError with
# the name of its type, because the client can lack its module, e.g. zstandard
EXCEPTION_MODULES = frozenset({"artistools", "builtins", "numpy", "polars"})

# the classes that a reply can build with no side effect. numpy has classes with side effects, e.g. numpy.memmap
# writes a file, thus only these ones of numpy pass. The frames of polars go as tuples of Arrow IPC data, see
# FRAME_MARKER, because the __setstate__ of a polars frame unpickles a plan with the plain unpickler
REPLY_CLASSES = frozenset({
    ("argparse", "Namespace"),
    ("builtins", "bytearray"),
    ("builtins", "complex"),
    ("builtins", "frozenset"),
    ("builtins", "set"),
    ("builtins", "tuple"),
    ("collections", "OrderedDict"),
    ("datetime", "date"),
    ("datetime", "datetime"),
    ("datetime", "timedelta"),
    ("datetime", "timezone"),
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
})

# the functions that a reply can call. Every other object of a reply must be a class of a safe kind
REPLY_FUNCTIONS = frozenset({
    ("copyreg", "__newobj__"),
    ("copyreg", "_reconstructor"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy._core.numeric", "_frombuffer"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy.core.numeric", "_frombuffer"),
})

# a polars frame goes in a message as a plain tuple of this text, its Arrow IPC data, and whether it was a LazyFrame.
# The receiver makes the frame after the unpickle, see restore_frames
FRAME_MARKER = "artistools polars frame"

# the maximum size of a message, which limits the memory that a reply of a host can take
MAX_MESSAGE_BYTES = 16 * 1024**3

# the server writes this line before its first message. A startup file of the remote shell can write text to the
# standard output first, thus the client ignores each line before this one. Protocol 2 sends the version of polars
# and the standard error of a quiet request
SERVER_PROTOCOL_PREFIX = b"artistools server protocol "
SERVER_START_LINE = SERVER_PROTOCOL_PREFIX + b"2\n"

# each message starts with the length of its data in this number of bytes. One byte then shows whether zlib
# compressed the data, and the pickle data comes last. The receiver reads a full message before it unpickles it,
# thus after an unpickle error the next read starts at the next message
MESSAGE_LENGTH_BYTES = 8

# zlib makes a message of at least this size smaller before it goes through ssh. The emission contributions of a 3D
# model were 2.55 MB, and zlib level 1 made them 0.25 MB in 9 ms. For a smaller message, zlib takes more time than it
# saves in the transfer
COMPRESS_MIN_BYTES = 64 * 1024

# polars starts one thread for each core, and a login node of a cluster has hundreds of cores. On a node of 384 cores,
# 16 threads made a plot of estimators 0.9 s in place of 2.1 s, and a light curve from packets 1.1 s in place of 1.4 s
SERVER_POLARS_THREADS = 16


def split_remote_path(path: Path | str) -> tuple[str, Path] | None:
    """Return the host and the path on that host for a path of the form "host:path", or None for a local path.

    As for rsync, only the text decides, and the local files do not. ssh starts the server in the home folder, thus
    a relative path gets "~" at its start. The path is then absolute on the server, e.g. "vae26:" and "vae26:." name
    the home folder and not a folder with no name.
    """
    match = REMOTEPATH_PATTERN.match(str(path))
    if match is None:
        return None

    hostpath = Path(match["path"])
    if not hostpath.is_absolute() and not match["path"].startswith("~"):
        hostpath = Path("~", hostpath)

    return match["host"], hostpath


def is_remote_path(path: Path | str) -> bool:
    """Return whether the path names a folder or a file on a different host."""
    return split_remote_path(path) is not None


def model_path_from_text(text: str) -> Path:
    """Return the Path of a model path that a user writes.

    Path removes the "./" at the start of "./run:2", and the colon of the result then makes a remote path. Thus a
    local path with a colon before its first slash becomes absolute. argparse uses this function as the type of a
    model path.
    """
    path = Path(text)
    return path.absolute() if text.startswith("./") and is_remote_path(path) else path


def get_canonical_path(path: Path) -> Path:
    """Return a remote path with its relative part below "~", or a local path unchanged.

    The parent of Path("vae26:light_curve.out") is Path("."), which is a local folder. The parent of
    Path("vae26:~/light_curve.out") keeps the host.
    """
    remote = split_remote_path(path)
    return path if remote is None else Path(f"{remote[0]}:{remote[1]}")


def check_local_path(path: Path | str) -> None:
    """Stop with an error when a reader opens a remote path on the local host.

    Only a reader with the decorator on_model_host reads a remote model. A different reader would give
    FileNotFoundError, and a caller can take that error as a file that does not exist. The caller then
    leaves out a part of the plot with no message. Path removes the "./" of a local path such as "./obs:curve.txt",
    thus a local file that exists passes.
    """
    if is_remote_path(path) and not Path(path).exists():
        msg = (
            f"artistools cannot read {path} with this option, because the file is on a different host."
            " Run the command on that host"
        )
        raise ValueError(msg)


def names_a_remote_folder(text: str) -> bool:
    """Return whether a word of the command line names a folder on a different host.

    A label such as "second:label" has the form host:path, thus only a path that starts with "~", or that holds a
    "/", counts. A label can also hold a space, e.g. "W7: 56Ni/56Co", and a remote folder seldom does. Thus a word
    with a space does not count. Such a remote folder, and a relative remote path such as "vae26:model", must then
    come before the options.
    """
    remote = REMOTEPATH_PATTERN.match(text)
    return (
        remote is not None
        and re.search(r"\s", text) is None
        and (remote["path"].startswith("~") or "/" in remote["path"])
    )


def is_plain_value(value: t.Any) -> bool:
    """Return whether a value of an argparse.Namespace can go to the server.

    A plain value is one of these:

    - None, a bool, a number, a str, or a Path;
    - a list or a tuple of plain values;
    - a dict with plain keys and plain values.

    The Namespace of a command also holds the parser, the function of the command, and with --quiet a stream. The
    server needs none of these, and pickle cannot send all of them.
    """
    if value is None or isinstance(value, (bool, int, float, str, Path)):
        return True
    if type(value) in {list, tuple}:
        return all(is_plain_value(item) for item in value)
    if type(value) is dict:
        return all(is_plain_value(key) and is_plain_value(item) for key, item in value.items())
    return False


def map_leaves(value: t.Any, func: Callable[[t.Any], t.Any]) -> t.Any:
    """Apply the function to each item inside the lists, the tuples, and the dict values of the value.

    Only these exact types hold the paths of an argument or a result. Examples are a list of model paths and a dict
    of LazyFrames for each direction bin. A NamedTuple or a different class stays as it is.
    """
    if type(value) is list:
        return [map_leaves(item, func) for item in value]
    if type(value) is tuple:
        return tuple(map_leaves(item, func) for item in value)
    if type(value) is dict:
        return {key: map_leaves(item, func) for key, item in value.items()}
    return func(value)


def to_server_arguments(value: t.Any) -> tuple[str | None, t.Any]:
    """Return the host of the remote paths in the value, and the value with the path on that host for each one.

    The host is None if all the paths are local. One call runs on one host, thus the paths of two hosts give an
    error.
    """
    hosts: set[str] = set()

    def to_server_path(leaf: t.Any) -> t.Any:
        if isinstance(leaf, Path) and (remote := split_remote_path(leaf)) is not None:
            hosts.add(remote[0])
            return remote[1]
        if isinstance(leaf, argparse.Namespace):
            # the reader gets its model path as an argument of its own. The paths of the Namespace, e.g. the models
            # of a band plot on two hosts, thus give no host to the call, and the server does not use them
            return argparse.Namespace(**{key: value for key, value in vars(leaf).items() if is_plain_value(value)})
        return leaf

    servervalue = map_leaves(value, to_server_path)
    if len(hosts) > 1:
        msg = f"One function cannot read the models of two hosts: {', '.join(sorted(hosts))}"
        raise ValueError(msg)

    return next(iter(hosts), None), servervalue


def get_ipc_bytes(df: "pl.DataFrame") -> bytes:
    """Return the Arrow IPC data of a DataFrame."""
    import io

    buffer = io.BytesIO()
    df.write_ipc(buffer)
    return buffer.getvalue()


def reduce_dataframe(df: "pl.DataFrame") -> tuple[type[tuple[t.Any, ...]], tuple[tuple[str, bytes, bool]]]:
    """Return the pickle form of a DataFrame: a plain tuple that holds FRAME_MARKER and the Arrow IPC data."""
    return tuple, ((FRAME_MARKER, get_ipc_bytes(df), False),)


def reduce_lazyframe(lf: "pl.LazyFrame") -> tuple[type[tuple[t.Any, ...]], tuple[tuple[str, bytes, bool]]]:
    """Return the pickle form of a LazyFrame, which holds the Arrow IPC data of its result.

    The plan of a LazyFrame reads the files of the server, thus the client cannot collect it.
    """
    return tuple, ((FRAME_MARKER, get_ipc_bytes(lf.collect()), True),)


def restore_frames(value: t.Any) -> t.Any:
    """Return the value of a message with a polars frame in place of each tuple of FRAME_MARKER."""
    import io

    import polars as pl

    if type(value) is tuple and len(value) == 3 and value[0] == FRAME_MARKER and isinstance(value[1], bytes):
        df = pl.read_ipc(io.BytesIO(value[1]))
        return df.lazy() if value[2] else df
    if type(value) is list:
        return [restore_frames(item) for item in value]
    if type(value) is tuple:
        return tuple(restore_frames(item) for item in value)
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        # a NamedTuple of a result, e.g. the data of a figure, can hold frames
        return type(value)(*(restore_frames(item) for item in value))
    if type(value) is dict:
        return {key: restore_frames(item) for key, item in value.items()}
    return value


def dump_message(value: t.Any) -> bytes:
    """Return the pickle data of a message.

    The pickle of a polars frame holds a format that changes between two versions of polars. uvx can give the server
    a different polars than the client, thus a frame goes as Arrow IPC data, which does not change.
    """
    import copyreg
    import io

    import polars as pl

    buffer = io.BytesIO()
    pickler = pickle.Pickler(buffer, protocol=pickle.HIGHEST_PROTOCOL)
    pickler.dispatch_table = copyreg.dispatch_table | {pl.DataFrame: reduce_dataframe, pl.LazyFrame: reduce_lazyframe}
    pickler.dump(value)
    return buffer.getvalue()


def write_message(stream: t.IO[bytes], data: bytes) -> None:
    """Write the length of the data, the flag of the compression, and then the data."""
    import zlib

    compressed = len(data) >= COMPRESS_MIN_BYTES
    if compressed:
        data = zlib.compress(data, 1)
    stream.write(len(data).to_bytes(MESSAGE_LENGTH_BYTES, "big") + bytes([compressed]))
    stream.write(data)
    stream.flush()


def is_reply_exception(exc: BaseException) -> bool:
    """Return whether the client accepts the exception in a reply.

    SystemExit comes from exit_with_error on the server, and it stops the command of the client too. KeyboardInterrupt
    and GeneratorExit are not errors of a reader.
    """
    if type(exc).__module__.split(".", maxsplit=1)[0] not in EXCEPTION_MODULES:
        return False
    return isinstance(exc, Exception) or type(exc) is SystemExit


def is_reply_class(cls: type, module: str) -> bool:
    """Return whether a reply can build the class: a NamedTuple of a result, a path, or an exception of a reader."""
    if issubclass(cls, tuple):
        return module.startswith("artistools.")
    if issubclass(cls, Path):
        return module.startswith("pathlib")
    return issubclass(cls, BaseException) and module.split(".", maxsplit=1)[0] in EXCEPTION_MODULES


class ReplyUnpickler(pickle.Unpickler):
    """Unpickle a reply of the server, with only the objects that a result or an exception needs.

    A plain unpickler calls any function that the data names. A different user can control a remote host, and the
    data would then run a command on this host.
    """

    @t.override
    def find_class(self, module: str, name: str, /) -> t.Any:
        """Return an allowed class or function of a module that this process has loaded, and refuse each other object.

        The import of a module runs its code, thus the unpickler imports no module. A name with a dot reads an
        attribute of a different object, thus the unpickler refuses it.
        """
        import sys

        loadedmodule = sys.modules.get(module)
        found = None if "." in name or loadedmodule is None else getattr(loadedmodule, name, None)
        if module.split(".", maxsplit=1)[0] in REPLY_MODULES | EXCEPTION_MODULES and found is not None:
            if (module, name) in REPLY_FUNCTIONS or (module, name) in REPLY_CLASSES:
                return found
            if isinstance(found, type) and is_reply_class(found, module):
                return found
        msg = f"A reply of the artistools server holds {module}.{name}, which the client does not accept"
        raise pickle.UnpicklingError(msg)


def load_reply(data: bytes) -> t.Any:
    """Return the value of a reply of the server, with the polars frames that it holds."""
    import io

    return restore_frames(ReplyUnpickler(io.BytesIO(data)).load())


def read_message(stream: t.IO[bytes]) -> bytes:
    """Return the pickle data of the next message. Raise EOFError if the stream ends first."""
    import zlib

    header = stream.read(MESSAGE_LENGTH_BYTES + 1)
    if len(header) < MESSAGE_LENGTH_BYTES + 1:
        raise EOFError
    size = int.from_bytes(header[:MESSAGE_LENGTH_BYTES], "big")
    if size > MAX_MESSAGE_BYTES:
        msg = f"A message of {size} bytes is larger than the maximum of {MAX_MESSAGE_BYTES} bytes"
        raise ValueError(msg)
    data = stream.read(size)
    if len(data) < size:
        raise EOFError
    if not header[MESSAGE_LENGTH_BYTES]:
        return data
    decompressor = zlib.decompressobj()
    data = decompressor.decompress(data, MAX_MESSAGE_BYTES)
    if decompressor.unconsumed_tail:
        msg = f"A message holds more than the maximum of {MAX_MESSAGE_BYTES} bytes"
        raise ValueError(msg)
    return data


def get_polars_pin() -> str:
    """Return the option of uvx that gives the server the polars of this process.

    The filter of a query of a remote model goes to the host as a serialised polars expression, see
    read_remote_estimator_rows. The format of such an expression changes between two versions of polars.
    """
    import polars as pl

    return f"--with polars=={pl.__version__}"


def get_server_argv(host: str, command: str) -> list[str]:
    """Return the command line that starts the artistools server on the host with the command.

    ssh gives the command to the shell of the remote host, thus the command is one string.
    """
    if host.startswith("-"):
        msg = f"The host {host} starts with -, and ssh would read it as an option"
        raise ValueError(msg)
    # ssh takes an IPv6 address with no brackets
    sshhost = re.sub(r"\[([^\]]*)\]", r"\1", host)
    return ["ssh", "--", sshhost, command]


def get_release_command() -> str:
    """Return the command that runs the release of this version of artistools with uvx, thus the host needs only uv."""
    from importlib.metadata import version

    return (
        f"POLARS_MAX_THREADS={SERVER_POLARS_THREADS} uvx {get_polars_pin()} artistools@{version('artistools')} server"
    )


def get_git_command(gitsource: "GitSource") -> str:
    """Return the command that runs the git commit of this artistools with uvx. The host needs git and Rust."""
    return (
        f"POLARS_MAX_THREADS={SERVER_POLARS_THREADS} uvx {get_polars_pin()}"
        f' --from "artistools @ git+{gitsource.url}@{gitsource.commit}" artistools server'
    )


def choose_server_command(givencommand: str | None, gitsource: "GitSource | None") -> tuple[str, str]:
    """Return the command that starts the server on a host, and the reason for the choice.

    A reader must have the same code on the two sides. The function selects the first command that applies:
    - the value of ARTISTOOLS_REMOTE_COMMAND, because it is the choice of the user;
    - the release of this version, if this artistools is a release or has the package code of the release;
    - the release of this version, if the commit is on no remote branch, because the remote host cannot get it;
    - the git commit, if its package code differs from the release, or if the clone has no tag of the release.
    The release needs no build on the host. An empty ARTISTOOLS_REMOTE_COMMAND gives the automatic choice, because ssh
    starts a login shell for an empty command.
    """
    from importlib.metadata import version

    localversion = version("artistools")
    if givencommand:
        return givencommand, f"{SERVER_COMMAND_ENVVAR} gives the command"
    if gitsource is None:
        return get_release_command(), f"this artistools is the release {localversion}"
    if gitsource.releasechanges == []:
        return (
            get_release_command(),
            f"the commit {gitsource.commit} has the same package code as the release {localversion}",
        )
    if not gitsource.ispushed:
        return (
            get_release_command(),
            f"the commit {gitsource.commit} is on no remote branch, thus the remote host cannot get it",
        )
    difference = (
        f"no release tag v{localversion} is in this clone to compare with"
        if gitsource.releasechanges is None
        else f"{len(gitsource.releasechanges)} files of the package differ from the release {localversion}"
    )
    return get_git_command(gitsource), difference


def read_server_versions(process: "subprocess.Popen[bytes]") -> tuple[str, str]:
    """Return the versions of artistools and polars that the server sends after its start line.

    Raise EOFError if the server stops first, and ConnectionError for the start line of a different protocol. That
    server then waits for a request, thus a search for the next line would wait for ever.
    """
    assert process.stdout is not None
    while not (line := process.stdout.readline(64 * 1024)).endswith(SERVER_START_LINE):
        if not line:
            raise EOFError
        if SERVER_PROTOCOL_PREFIX in line:
            serverprotocol = line[line.index(SERVER_PROTOCOL_PREFIX) :].strip().decode(errors="replace")
            msg = f"uses '{serverprotocol}', and this artistools needs '{SERVER_START_LINE.strip().decode()}'"
            raise ConnectionError(msg)

    versions = load_reply(read_message(process.stdout))
    assert isinstance(versions, tuple)
    assert len(versions) == 2
    assert all(isinstance(serverversion, str) for serverversion in versions)
    return versions


class GitSource(t.NamedTuple):
    """The git commit of this artistools, and the data that tells if a remote host can install that commit."""

    url: str
    commit: str
    # a host can get only a commit that is on a remote branch
    ispushed: bool
    # the package files that differ from the release of this version, or None if the clone has no tag to compare with
    releasechanges: list[str] | None
    notes: list[str]


def get_git_source() -> GitSource | None:
    """Return the git source of this artistools. Return None for a release.

    An install from git, e.g. with uvx --from git+https://..., records the commit in direct_url.json (PEP 610). An
    install of a clone records its folder there. git then gives the commit of the folder of the code that runs, which
    can be a different clone, e.g. with PYTHONPATH.
    """
    import json
    import subprocess  # ruff:ignore[suspicious-subprocess-import]
    from importlib.metadata import distribution
    from importlib.metadata import version

    directurl = distribution("artistools").read_text("direct_url.json")
    if directurl is None:
        return None

    source = json.loads(directurl)
    if (vcsinfo := source.get("vcs_info")) is not None:
        # a host can get the commit of an install from git, and this install has no tag to compare with
        if vcsinfo.get("vcs") != "git":
            return None
        return GitSource(source["url"], vcsinfo["commit_id"], ispushed=True, releasechanges=None, notes=[])

    folder = str(Path(__file__).resolve().parents[2])

    def run_git(*gitargs: str) -> str:
        return subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
            ["git", "-C", folder, *gitargs],  # ruff:ignore[start-process-with-partial-path]
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    try:
        commit = run_git("rev-parse", "HEAD")
        # the first remote branch that holds the commit, e.g. "origin/mybranch", names the remote of a fork too
        remotebranches = run_git("branch", "--remotes", "--contains", commit).split()
        url = run_git("remote", "get-url", remotebranches[0].split("/")[0]) if remotebranches else REPOSITORY_URL
        haschanges = bool(run_git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        # a folder that is not a clone, or a host with no git, gives no commit
        return None

    # the server does not run the tests, thus a change to a test file cannot change a reader
    releasechanges: list[str] | None
    try:
        changedfiles = run_git("diff", "--name-only", f"v{version('artistools')}", commit, "--", "artistools", "rust")
    except subprocess.CalledProcessError:
        # a version with no release yet, or a clone with no tags, has no tag to compare with
        releasechanges = None
    else:
        releasechanges = [
            path for path in changedfiles.splitlines() if not Path(path).name.startswith(("test_", "conftest"))
        ]

    # uv reads an ssh URL of git only in the form ssh://, and not in the form of scp, e.g. git@github.com:org/repo.git
    if (scpurl := re.match(r"^(?P<userhost>[^/:]+@[^/:]+):(?P<path>.*)$", url)) is not None:
        url = f"ssh://{scpurl['userhost']}/{scpurl['path']}"

    notes = []
    if not remotebranches:
        notes.append("Push the commit to GitHub first. The host cannot get a commit that is only on this host.")
    if haschanges:
        notes.append("The changes that are not in a commit stay on this host.")

    return GitSource(url, commit, ispushed=bool(remotebranches), releasechanges=releasechanges, notes=notes)


def get_git_server_suggestion(host: str, givencommand: str | None, gitsource: GitSource) -> str:
    """Return the text that tells the user how to run the commit of this artistools on the host.

    givencommand is the value of ARTISTOOLS_REMOTE_COMMAND, or None if the release runs on the host. A given command
    can name an old commit, thus the text also tells whether it names the commit of this artistools.
    """
    if givencommand is None:
        commandtext = f"but the server command runs a release. To run the same commit on {host}, set:"
    elif gitsource.commit in givencommand:
        commandtext = f"and {SERVER_COMMAND_ENVVAR} names this commit. The usual command for this commit is:"
    else:
        commandtext = f"and {SERVER_COMMAND_ENVVAR} does not name this commit. To run the same commit on {host}, set:"
    return "\n".join([
        f"This artistools comes from the git commit {gitsource.commit}, {commandtext}",
        f"  export {SERVER_COMMAND_ENVVAR}='{get_git_command(gitsource)}'",
        "The first start builds the Rust extension of artistools on the host, thus the host needs git and Rust.",
        *gitsource.notes,
    ])


def get_server(host: str) -> "tuple[subprocess.Popen[bytes], threading.Lock]":
    """Return the process and the pipe lock of the artistools server on the host. Start the server if necessary.

    The server stops when this process closes the pipes at its exit.
    """
    with SERVERS_LOCK:
        if (host, os.getpid()) not in SERVERS:
            SERVERS[host, os.getpid()] = start_server(host)
        return SERVERS[host, os.getpid()]


def forget_server(host: str, process: "subprocess.Popen[bytes] | None" = None) -> None:
    """Remove the server of the host from the registry. The next call starts a new server.

    With a process, only that server leaves the registry. A thread can fail on a server that a different thread
    replaced, and the new server must then stay.
    """
    with SERVERS_LOCK:
        entry = SERVERS.get((host, os.getpid()))
        if entry is not None and (process is None or entry[0] is process):
            del SERVERS[host, os.getpid()]


def close_server_pipes(process: "subprocess.Popen[bytes]") -> None:
    """Close the pipes of a server, and wait a short time for its exit. The server stops at the end of its input."""
    import contextlib
    import subprocess  # ruff:ignore[suspicious-subprocess-import]

    with contextlib.suppress(OSError):
        if process.stdin is not None:
            process.stdin.close()
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=5)


def launch_server(argv: list[str]) -> "tuple[subprocess.Popen[bytes], str, str]":
    """Start a server with the command line, and return the process and the artistools and polars versions of it.

    If the start fails, the function stops the process and raises the error of read_server_versions again.
    """
    import subprocess  # ruff:ignore[suspicious-subprocess-import]

    process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE)  # ruff:ignore[subprocess-without-shell-equals-true]
    try:
        serverversion, serverpolarsversion = read_server_versions(process)
    except BaseException:
        process.kill()
        raise
    return process, serverversion, serverpolarsversion


def start_server(host: str) -> "tuple[subprocess.Popen[bytes], threading.Lock]":
    """Start the artistools server on the host, and return the process and the lock of its pipes.

    The server stays for the life of the process, e.g. for each plot of a viewer, because a start through ssh takes
    about 10 s. The exit of the process closes its pipes, and the server then stops. choose_server_command gives the
    command. If the server of a git commit does not start, e.g. because the host has no Rust, the release starts.
    """
    import atexit
    import shlex
    from importlib.metadata import version

    import polars as pl

    from artistools.misc.cliutils import exit_with_error
    from artistools.misc.cliutils import print_detail
    from artistools.misc.cliutils import print_warning

    localversion = version("artistools")
    givencommand = os.environ.get(SERVER_COMMAND_ENVVAR) or None
    gitsource = get_git_source()
    command, reason = choose_server_command(givencommand, gitsource)
    runscommit = gitsource is not None and givencommand is None and command == get_git_command(gitsource)
    # the server can run different code from this artistools, thus the warning tells how to run the same commit
    showssuggestion = (
        gitsource is not None and not runscommit and (givencommand is not None or gitsource.releasechanges != [])
    )
    if showssuggestion:
        assert gitsource is not None
        print_warning(get_git_server_suggestion(host, givencommand, gitsource))
    elif runscommit and gitsource is not None and gitsource.notes:
        print_warning("\n".join(gitsource.notes))
    starthelp = (
        f"Set {SERVER_COMMAND_ENVVAR} as the warning above shows"
        if showssuggestion
        else f"Install uv on {host}. The default command needs a release {localversion} of artistools with the"
        f" server command. As an alternative, set {SERVER_COMMAND_ENVVAR} to a command that starts such a server"
    )

    def get_start_error(argv: list[str], exc: Exception) -> str:
        if isinstance(exc, ConnectionError):
            return f"the artistools server on {host} {exc}"
        # text of a shell can give any error of the unpickle
        return f"the artistools server on {host} did not start. The ssh command line was: {shlex.join(argv)}"

    argv = get_server_argv(host, command)
    print_detail(f"artistools starts the server on {host} with: {shlex.join(argv)}")
    print_detail(f"Reason: {reason}.")
    if runscommit:
        print_detail("The first start of a commit builds the Rust extension of artistools on the host, thus it is slow")
    try:
        process, serverversion, serverpolarsversion = launch_server(argv)
    except Exception as exc:  # ruff:ignore[blind-except]
        if not runscommit:
            exit_with_error(get_start_error(argv, exc), starthelp)
        print_warning(
            f"The server of the commit did not start on {host}. To build the commit, the host needs git and Rust."
            f" artistools starts the release {localversion} instead, and the release possibly reads some files"
            " differently from this commit"
        )
        argv = get_server_argv(host, get_release_command())
        print_detail(f"artistools starts the server on {host} with: {shlex.join(argv)}")
        try:
            process, serverversion, serverpolarsversion = launch_server(argv)
        except Exception as releaseexc:  # ruff:ignore[blind-except]
            exit_with_error(get_start_error(argv, releaseexc), starthelp)

    atexit.register(close_server_pipes, process)
    if serverpolarsversion != pl.__version__:
        process.kill()
        exit_with_error(
            f"the artistools server on {host} has polars {serverpolarsversion}, and this artistools has polars"
            f" {pl.__version__}. The server reads the polars expressions of the client, and these expressions need the"
            " same version of polars",
            f"Add {get_polars_pin()} to the uvx command of {SERVER_COMMAND_ENVVAR}",
        )
    if serverversion != localversion:
        print_warning(
            f"The artistools server on {host} has version {serverversion}, but this artistools has version "
            f"{localversion}. A reader with a different signature on the server gives an error"
        )

    return process, threading.Lock()


def output_is_hidden() -> bool:
    """Return whether the standard output of the calling thread goes elsewhere than to the terminal of the process.

    --quiet sends it to the null device, and the worker thread of a viewer sends it to a buffer. The server then
    hides the standard output of the reader, and it sends back the standard error, e.g. an error or a warning.
    The ThreadOutput of a viewer gives the target of each thread.
    """
    import sys

    stream: object = sys.stdout
    if (get_target := getattr(stream, "get_target", None)) is not None:
        stream = get_target()
    return stream is not sys.__stdout__


def call_on_host(host: str, modulename: str, qualname: str, args: tuple[t.Any, ...], kwargs: dict[str, t.Any]) -> t.Any:
    """Run the function on the artistools server of the host, and return its result."""
    import sys

    process, lock = get_server(host)
    assert process.stdin is not None
    assert process.stdout is not None

    quiet = output_is_hidden()
    request = dump_message((modulename, qualname, args, kwargs, quiet))
    with lock:
        try:
            write_message(process.stdin, request)
            response = read_message(process.stdout)
        except BaseException as exc:
            # an exchange that stops in the middle, e.g. at Ctrl-C, leaves a part of a message in the pipes. The next
            # call would read it, thus the server stops, and the next call starts a new one
            process.kill()
            forget_server(host, process)
            if isinstance(exc, (BrokenPipeError, EOFError)):
                msg = f"The artistools server on {host} stopped during a call of {qualname}"
                raise OSError(msg) from exc
            raise

    succeeded, result, errortext = load_reply(response)
    # a viewer shows the first line of an error and the last warning of the buffer of its thread
    sys.stderr.write(errortext)
    if not succeeded:
        assert isinstance(result, BaseException)
        result.add_note(f"The error came from {qualname} on the artistools server of {host}")
        raise result

    # the server gives back the paths of its own files, thus each one gets the name of the host
    return map_leaves(result, lambda leaf: Path(f"{host}:{leaf}") if isinstance(leaf, Path) else leaf)


def on_model_host[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Run the function on the host of the model when an argument is a remote path.

    The decorator finds a remote path only in a Path argument. A str argument stays local, because a label can have
    the form "name:text". The arguments go through pickle, and an argparse.Namespace goes with only its plain values
    (see is_plain_value). A LazyFrame result comes back as the collected data.

    Put this decorator only on a function that returns a small result, e.g. a spectrum. Do not put it on a function
    that returns all the packets.
    """

    @functools.wraps(func)
    def run_on_model_host(*args: P.args, **kwargs: P.kwargs) -> R:
        host, (serverargs, serverkwargs) = to_server_arguments((args, kwargs))
        if host is None:
            return func(*args, **kwargs)

        return t.cast(
            "R", call_on_host(host, func.__module__, run_on_model_host.__qualname__, serverargs, serverkwargs)
        )

    return run_on_model_host


def get_server_function(modulename: str, qualname: str) -> Callable[..., t.Any]:
    """Return the function that a client names by its module and its qualified name."""
    import importlib

    if modulename.split(".", maxsplit=1)[0] != "artistools":
        msg = f"The artistools server runs only the functions of artistools, and not {modulename}.{qualname}"
        raise ValueError(msg)

    func: t.Any = importlib.import_module(modulename)
    for name in qualname.split("."):
        func = getattr(func, name)
    # the client caches the result of a remote call. A cache of the server would keep old data after a reload of a
    # viewer, thus the server calls the function below its lru_cache
    while hasattr(func, "cache_clear") and hasattr(func, "__wrapped__"):
        func = func.__wrapped__

    return t.cast("Callable[..., t.Any]", func)


def expand_home(leaf: t.Any) -> t.Any:
    """Replace the "~" at the start of a path with the home folder of the server."""
    return leaf.expanduser() if isinstance(leaf, Path) else leaf


def collect_lazyframes(value: t.Any) -> t.Any:
    """Return the value with each LazyFrame in its lists, tuples, and dict values collected in one batch.

    The frames of a reader can share a scan, e.g. the 100 direction bins of light_curve_res.out. pl.collect_all
    reads that file once, but a collect of each frame reads it once for each frame.
    """
    import polars as pl

    lazyframes: list[pl.LazyFrame] = []

    def add_lazyframe(leaf: t.Any) -> None:
        if isinstance(leaf, pl.LazyFrame):
            lazyframes.append(leaf)

    map_leaves(value, add_lazyframe)
    if not lazyframes:
        return value

    collected = iter(pl.collect_all(lazyframes))

    def to_collected(leaf: t.Any) -> t.Any:
        return next(collected).lazy() if isinstance(leaf, pl.LazyFrame) else leaf

    return map_leaves(value, to_collected)


def run_function(request: tuple[str, str, tuple[t.Any, ...], dict[str, t.Any], bool], errorstream: t.TextIO) -> t.Any:
    """Return the result of the function of an unpickled request.

    With quiet, the function prints nothing to the standard output, and its standard error goes to errorstream.
    """
    import contextlib

    modulename, qualname, args, kwargs, quiet = request
    func = get_server_function(modulename, qualname)
    with contextlib.ExitStack() as stack:
        if quiet:
            devnull = stack.enter_context(Path(os.devnull).open("w", encoding="utf-8"))
            stack.enter_context(contextlib.redirect_stdout(devnull))
            stack.enter_context(contextlib.redirect_stderr(errorstream))
        return collect_lazyframes(func(*map_leaves(args, expand_home), **map_leaves(kwargs, expand_home)))


def run_request(request: bytes) -> bytes:
    """Run one request of a client, and return the message with the result or the exception, and the error text.

    The error text holds the standard error of a quiet request, which the client writes to its own standard error.
    """
    import contextlib
    import io
    import traceback

    errorstream = io.StringIO()
    try:
        return dump_message((
            True,
            run_function(restore_frames(pickle.loads(request)), errorstream),
            errorstream.getvalue(),
        ))
    except BaseException as exc:
        # a panic of polars or of rustext is a BaseException. The client gets it as an error, and the server keeps
        # its state for the next request
        if isinstance(exc, KeyboardInterrupt):
            raise
        # the client raises the same exception, thus a caller that catches FileNotFoundError still works. The client
        # shows the traceback of the server with ARTISTOOLS_TRACEBACK=1 only, as for a local error
        exc.add_note("The traceback on the server:\n" + "".join(traceback.format_exception(exc)).rstrip())
        if is_reply_exception(exc):
            with contextlib.suppress(Exception):
                return dump_message((False, exc, errorstream.getvalue()))
        msg = f"{type(exc).__name__}: {exc}"
        replyexc = RuntimeError(msg)
        for note in getattr(exc, "__notes__", []):
            replyexc.add_note(note)
        return dump_message((False, replyexc, errorstream.getvalue()))


def serve(requeststream: t.IO[bytes], resultstream: t.IO[bytes]) -> None:
    """Run each request of the request stream, and write each result to the result stream."""
    from importlib.metadata import version

    import polars as pl

    resultstream.write(SERVER_START_LINE)
    write_message(resultstream, dump_message((version("artistools"), pl.__version__)))
    while True:
        try:
            request = read_message(requeststream)
        except EOFError:
            return

        write_message(resultstream, run_request(request))


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add no arguments. A client starts the server, and the server reads its requests from the standard input."""


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Run the requests that an artistools client sends through ssh. A path host:path starts this server."""
    import io
    import sys

    from artistools.misc.cliutils import parse_cli_args

    parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    # the results use the standard output, thus the messages of a reader go to the standard error, which the client
    # shows to the user. The file descriptors are the ones of the process, because --quiet replaces sys.stdout
    with os.fdopen(os.dup(1), "wb") as resultstream:
        os.dup2(2, 1)
        # the standard output was a pipe at the start, thus Python gave it a large buffer. The user then saw each
        # message of a reader only when the buffer was full
        if isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout.reconfigure(line_buffering=True)

        serve(sys.stdin.buffer, resultstream)
    # the client reports the time of its command, thus the server stops with no report of its own
    raise SystemExit(0)
