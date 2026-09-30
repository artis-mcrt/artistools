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

# the rule of rsync: a colon before the first slash makes a remote path, e.g. "vae26:~/mymodel" and
# "user@vae26:/lustre/mymodel". A local path with such a colon starts with "./", e.g. "./run:2". An IPv6 address
# is in brackets, e.g. "user@[2001:db8::1]:/lustre/mymodel"
REMOTEPATH_PATTERN = re.compile(r"^(?P<host>(?:[^/:@\[]*@)?\[[^\]/]*\]|[^/:\[]+):(?P<path>.*)$", re.DOTALL)

# a user can give a different command to start the server, e.g. the path of an artistools in a clone
SERVER_COMMAND_ENVVAR = "ARTISTOOLS_REMOTE_COMMAND"

# the repository that a host can get a commit of a clone from
REPOSITORY_URL = "https://github.com/artis-mcrt/artistools"

# the servers of this process, with the lock of the pipes of each one. The key holds the process ID, because a
# child process cannot use the pipes of its parent. A failed server leaves the registry alone, and the servers of
# the other hosts stay
SERVERS: "dict[tuple[str, int], tuple[subprocess.Popen[bytes], threading.Lock]]" = {}
# two threads can make the first call to one host at the same time. The lock makes the start of its server happen
# once
SERVERS_LOCK = threading.Lock()

# the modules of the objects that a reply of the server can hold. The unpickler of the client imports nothing else
REPLY_MODULES = frozenset({
    "argparse",
    "artistools",
    "builtins",
    "collections",
    "copyreg",
    "datetime",
    "json",
    "numpy",
    "pathlib",
    "pickle",
    "polars",
    "pyo3_runtime",
    "subprocess",
    "zstandard",
})

# the functions that a reply can call. Every other object of a reply must be a class of a safe kind
REPLY_FUNCTIONS = frozenset({
    ("artistools.misc.remote", "dataframe_from_ipc"),
    ("artistools.misc.remote", "lazyframe_from_ipc"),
    ("copyreg", "__newobj__"),
    ("copyreg", "_reconstructor"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy._core.numeric", "_frombuffer"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy.core.numeric", "_frombuffer"),
})

# the server writes this line before its first message. A startup file of the remote shell can write text to the
# standard output first, thus the client ignores each line before this one
SERVER_START_LINE = b"artistools server protocol 1\n"

# each message is the length of its pickle data in this number of bytes, then the pickle data. The receiver reads
# a full message before it unpickles it, thus an error in the unpickle leaves the next message in step
MESSAGE_LENGTH_BYTES = 8


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

    A label such as "second:label" has the form host:path, thus only a path that starts with "~" or "/", or that
    holds a "/", counts. A relative remote path such as "vae26:model" must then come before the options.
    """
    remote = REMOTEPATH_PATTERN.match(text)
    return remote is not None and (remote["path"].startswith(("~", "/")) or "/" in remote["path"])


def is_plain_value(value: t.Any) -> bool:
    """Return whether a value of an argparse.Namespace goes to the server: None, a number, a str, a Path, or a list.

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
            plainvalues = {key: value for key, value in vars(leaf).items() if is_plain_value(value)}
            return argparse.Namespace(**map_leaves(plainvalues, to_server_path))
        return leaf

    servervalue = map_leaves(value, to_server_path)
    if len(hosts) > 1:
        msg = f"One function cannot read the models of two hosts: {', '.join(sorted(hosts))}"
        raise ValueError(msg)

    return next(iter(hosts), None), servervalue


def dataframe_from_ipc(data: bytes) -> "pl.DataFrame":
    """Return the DataFrame of Arrow IPC data."""
    import io

    import polars as pl

    return pl.read_ipc(io.BytesIO(data))


def lazyframe_from_ipc(data: bytes) -> "pl.LazyFrame":
    """Return a LazyFrame that holds the DataFrame of Arrow IPC data."""
    return dataframe_from_ipc(data).lazy()


def get_ipc_bytes(df: "pl.DataFrame") -> bytes:
    """Return the Arrow IPC data of a DataFrame."""
    import io

    buffer = io.BytesIO()
    df.write_ipc(buffer)
    return buffer.getvalue()


def reduce_dataframe(df: "pl.DataFrame") -> tuple[Callable[[bytes], "pl.DataFrame"], tuple[bytes]]:
    """Return the pickle form of a DataFrame, which holds its Arrow IPC data."""
    return dataframe_from_ipc, (get_ipc_bytes(df),)


def reduce_lazyframe(lf: "pl.LazyFrame") -> tuple[Callable[[bytes], "pl.LazyFrame"], tuple[bytes]]:
    """Return the pickle form of a LazyFrame, which holds the Arrow IPC data of its result.

    The plan of a LazyFrame reads the files of the server, thus the client cannot collect it.
    """
    return lazyframe_from_ipc, (get_ipc_bytes(lf.collect()),)


def dump_message(value: t.Any) -> bytes:
    """Return the pickle data of a message.

    The pickle of a polars frame holds a format that changes between two versions of polars. uvx can give the server
    a different polars than the client, thus a frame goes as Arrow IPC data, which does not change.
    """
    import copyreg
    import io
    import pickle

    import polars as pl

    buffer = io.BytesIO()
    pickler = pickle.Pickler(buffer, protocol=pickle.HIGHEST_PROTOCOL)
    pickler.dispatch_table = copyreg.dispatch_table | {pl.DataFrame: reduce_dataframe, pl.LazyFrame: reduce_lazyframe}
    pickler.dump(value)
    return buffer.getvalue()


def write_message(stream: t.IO[bytes], data: bytes) -> None:
    """Write the length of the pickle data and then the data."""
    stream.write(len(data).to_bytes(MESSAGE_LENGTH_BYTES, "big"))
    stream.write(data)
    stream.flush()


class ReplyUnpickler(pickle.Unpickler):
    """Unpickle a reply of the server, with only the objects that a result or an exception needs.

    A plain unpickler calls any function that the data names. A different user can control a remote host, and the
    data would then run a command on this host.
    """

    @t.override
    def find_class(self, module: str, name: str, /) -> t.Any:
        """Return a class of a safe kind or a function of REPLY_FUNCTIONS, and refuse each other object."""
        if module.split(".", maxsplit=1)[0] not in REPLY_MODULES:
            msg = f"A reply of the artistools server holds {module}.{name}, which the client does not accept"
            raise pickle.UnpicklingError(msg)
        found = super().find_class(module, name)
        if (module, name) in REPLY_FUNCTIONS:
            return found
        # a tuple class is a NamedTuple of a result. A reply builds the other classes with no call of a function
        # that has a side effect
        issafeclass = isinstance(found, type) and (
            issubclass(found, (BaseException, tuple, Path, frozenset, set, bytearray, complex, range, slice))
            or module.startswith(("numpy", "datetime", "collections", "argparse"))
        )
        if not issafeclass:
            msg = f"A reply of the artistools server holds {module}.{name}, which the client does not accept"
            raise pickle.UnpicklingError(msg)
        return found


def load_reply(data: bytes) -> t.Any:
    """Return the value of a reply of the server."""
    import io

    return ReplyUnpickler(io.BytesIO(data)).load()


def read_message(stream: t.IO[bytes]) -> bytes:
    """Return the pickle data of the next message. Raise EOFError if the stream ends first."""
    header = stream.read(MESSAGE_LENGTH_BYTES)
    if len(header) < MESSAGE_LENGTH_BYTES:
        raise EOFError
    size = int.from_bytes(header, "big")
    data = stream.read(size)
    if len(data) < size:
        raise EOFError
    return data


def get_server_argv(host: str) -> list[str]:
    """Return the command line that starts the artistools server on the host.

    The default command runs the release of this version with uvx, thus the host needs only uv. A reader
    must have the same signature on the two sides. ssh gives the command to the shell of the remote host,
    thus the command is one string. An empty ARTISTOOLS_REMOTE_COMMAND gives the default command, because ssh
    starts a login shell for an empty command.
    """
    import os
    from importlib.metadata import version

    defaultcommand = f"uvx artistools@{version('artistools')} server"
    # ssh takes an IPv6 address with no brackets
    sshhost = re.sub(r"\[([^\]]*)\]", r"\1", host)
    return ["ssh", sshhost, os.environ.get(SERVER_COMMAND_ENVVAR) or defaultcommand]


def read_server_version(process: "subprocess.Popen[bytes]") -> str:
    """Return the version that the server sends after its start line. Raise EOFError if the server stops first."""
    assert process.stdout is not None
    while not (line := process.stdout.readline()).endswith(SERVER_START_LINE):
        if not line:
            raise EOFError

    serverversion = load_reply(read_message(process.stdout))
    assert isinstance(serverversion, str)
    return serverversion


def get_git_source() -> tuple[str, str, list[str]] | None:
    """Return the git URL and the commit of this artistools, and notes on that commit. Return None for a release.

    An install from git, e.g. with uvx --from git+https://..., records the commit in direct_url.json (PEP 610). An
    install of a clone records its folder there. git then gives the commit of the folder of the code that runs, which
    can be a different clone, e.g. with PYTHONPATH.
    """
    import json
    import subprocess  # ruff:ignore[suspicious-subprocess-import]
    from importlib.metadata import distribution

    directurl = distribution("artistools").read_text("direct_url.json")
    if directurl is None:
        return None

    source = json.loads(directurl)
    if (vcsinfo := source.get("vcs_info")) is not None:
        return (source["url"], vcsinfo["commit_id"], []) if vcsinfo.get("vcs") == "git" else None

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

    # uv reads an ssh URL of git only in the form ssh://, and not in the form of scp, e.g. git@github.com:org/repo.git
    if (scpurl := re.match(r"^(?P<userhost>[^/:]+@[^/:]+):(?P<path>.*)$", url)) is not None:
        url = f"ssh://{scpurl['userhost']}/{scpurl['path']}"

    notes = []
    if not remotebranches:
        notes.append("Push the commit to GitHub first. The host cannot get a commit that is only on this host.")
    if haschanges:
        notes.append("The changes that are not in a commit stay on this host.")

    return url, commit, notes


def get_git_server_suggestion(host: str) -> str | None:
    """Return the text that tells the user how to run the commit of this artistools on the host, or None for a release.

    The default command runs the release of the same version, thus a clone with later commits runs different code
    on the server.
    """
    if (gitsource := get_git_source()) is None:
        return None

    url, commit, notes = gitsource
    servercommand = f'uvx --from "artistools @ git+{url}@{commit}" artistools server'
    return "\n".join([
        (
            f"This artistools comes from the git commit {commit}, but the default server command runs a release. To"
            f" run the same commit on {host}, set:"
        ),
        f"  export {SERVER_COMMAND_ENVVAR}='{servercommand}'",
        "The first start builds the Rust extension of artistools on the host, thus the host needs git and Rust.",
        *notes,
    ])


def get_server(host: str) -> "tuple[subprocess.Popen[bytes], threading.Lock]":
    """Return the process of the artistools server on the host, and the lock of its pipes. Start the server first.

    The server stops when this process closes the pipes at its exit.
    """
    import os

    with SERVERS_LOCK:
        if (host, os.getpid()) not in SERVERS:
            SERVERS[host, os.getpid()] = start_server(host)
        return SERVERS[host, os.getpid()]


def forget_server(host: str) -> None:
    """Remove the server of the host from the registry. The next call starts a new server."""
    import os

    with SERVERS_LOCK:
        SERVERS.pop((host, os.getpid()), None)


def start_server(host: str) -> "tuple[subprocess.Popen[bytes], threading.Lock]":
    """Start the artistools server on the host, and return the process and the lock of its pipes."""
    import os
    import shlex
    import subprocess  # ruff:ignore[suspicious-subprocess-import]
    from importlib.metadata import version

    from artistools.misc.cliutils import exit_with_error
    from artistools.misc.cliutils import print_detail
    from artistools.misc.cliutils import print_warning

    localversion = version("artistools")
    argv = get_server_argv(host)
    gitsuggestion = None if os.environ.get(SERVER_COMMAND_ENVVAR) else get_git_server_suggestion(host)
    if gitsuggestion is not None:
        print_warning(gitsuggestion)
    print_detail(f"artistools starts the server on {host} with: {shlex.join(argv)}")
    process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE)  # ruff:ignore[subprocess-without-shell-equals-true]

    try:
        serverversion = read_server_version(process)
    except Exception:  # ruff:ignore[blind-except]
        # text of a shell or a server of a different protocol can give any error of the unpickle
        process.kill()
        exit_with_error(
            f"the artistools server on {host} did not start. The ssh command line was: {shlex.join(argv)}",
            f"Set {SERVER_COMMAND_ENVVAR} as the warning above shows"
            if gitsuggestion is not None
            else f"Install uv on {host}. The default command needs a release {localversion} of artistools with the"
            f" server command. As an alternative, set {SERVER_COMMAND_ENVVAR} to a command that starts such a server",
        )

    if serverversion != localversion:
        print_warning(
            f"The artistools server on {host} has version {serverversion}, but this artistools has version "
            f"{localversion}. A reader with a different signature on the server gives an error"
        )

    return process, threading.Lock()


def call_on_host(host: str, modulename: str, qualname: str, args: tuple[t.Any, ...], kwargs: dict[str, t.Any]) -> t.Any:
    """Run the function on the artistools server of the host, and return its result."""
    import os
    import sys

    process, lock = get_server(host)
    assert process.stdin is not None
    assert process.stdout is not None

    # --quiet sends the standard output of the client to the null device, and the server then does the same
    quiet = getattr(sys.stdout, "name", None) == os.devnull
    request = dump_message((modulename, qualname, args, kwargs, quiet))
    with lock:
        try:
            write_message(process.stdin, request)
            response = read_message(process.stdout)
        except BaseException as exc:
            # an exchange that stops in the middle, e.g. at Ctrl-C, leaves a part of a message in the pipes. The next
            # call would read it, thus the server stops, and the next call starts a new one
            process.kill()
            forget_server(host)
            if isinstance(exc, (BrokenPipeError, EOFError)):
                msg = f"The artistools server on {host} stopped during a call of {qualname}"
                raise OSError(msg) from exc
            raise

    succeeded, result = load_reply(response)
    if not succeeded:
        assert isinstance(result, BaseException)
        result.add_note(f"The error came from {qualname} on the artistools server of {host}")
        raise result

    # the server gives back the paths of its own files, thus each one gets the name of the host
    return map_leaves(result, lambda leaf: Path(f"{host}:{leaf}") if isinstance(leaf, Path) else leaf)


def on_model_host[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Run the function on the host of the model when an argument is a remote path.

    A remote path is a Path object. A str argument stays local, because a label can have the form "name:text".
    The arguments go through pickle. An argparse.Namespace goes with its plain values only, see is_plain_value.
    The result must be small. A LazyFrame comes back as the collected data. Put this decorator on a
    function that reduces the data, e.g. a function that returns a spectrum. Do not put it on a function that
    returns all the packets.
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

    return t.cast("Callable[..., t.Any]", func)


def expand_home(leaf: t.Any) -> t.Any:
    """Replace the "~" at the start of a path with the home folder of the server."""
    return leaf.expanduser() if isinstance(leaf, Path) else leaf


def run_function(request: tuple[str, str, tuple[t.Any, ...], dict[str, t.Any], bool]) -> t.Any:
    """Return the result of the function of an unpickled request. With quiet, the function prints nothing."""
    import contextlib
    import os

    modulename, qualname, args, kwargs, quiet = request
    func = get_server_function(modulename, qualname)
    with contextlib.ExitStack() as stack:
        if quiet:
            devnull = stack.enter_context(Path(os.devnull).open("w", encoding="utf-8"))
            stack.enter_context(contextlib.redirect_stdout(devnull))
        return func(*map_leaves(args, expand_home), **map_leaves(kwargs, expand_home))


def run_request(request: bytes) -> bytes:
    """Run one request of a client, and return the message with the result or the exception."""
    import traceback

    try:
        return dump_message((True, run_function(pickle.loads(request))))
    except BaseException as exc:
        # a panic of polars or of rustext is a BaseException. The client gets it as an error, and the server keeps
        # its state for the next request
        if isinstance(exc, KeyboardInterrupt):
            raise
        # the client raises the same exception, thus a caller that catches FileNotFoundError still works. The client
        # shows the traceback of the server with ARTISTOOLS_TRACEBACK=1 only, as for a local error
        exc.add_note("The traceback on the server:\n" + "".join(traceback.format_exception(exc)).rstrip())
        try:
            return dump_message((False, exc))
        except Exception:  # ruff:ignore[blind-except]
            msg = f"{type(exc).__name__}: {exc}"
            return dump_message((False, RuntimeError(msg)))


def serve(requeststream: t.IO[bytes], resultstream: t.IO[bytes]) -> None:
    """Run each request of the request stream, and write each result to the result stream."""
    from importlib.metadata import version

    resultstream.write(SERVER_START_LINE)
    write_message(resultstream, dump_message(version("artistools")))
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
    import os
    import sys

    from artistools.misc.cliutils import parse_cli_args

    parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    # the results use the standard output, thus the messages of a reader go to the standard error, which the client
    # shows to the user. The file descriptors are the ones of the process, because --quiet replaces sys.stdout
    resultstream = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    # the standard output was a pipe at the start, thus Python gave it a large buffer. The user then saw each
    # message of a reader only when the buffer was full
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)

    serve(sys.stdin.buffer, resultstream)
    # the client reports the time of its command, thus the server stops with no report of its own
    raise SystemExit(0)
