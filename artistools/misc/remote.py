# ruff:file-ignore[suspicious-pickle-import, suspicious-pickle-usage]
"""Run the readers of a model on the host that holds the model.

A model path of the form "host:path" names a folder on a host that ssh reaches, e.g.
"vae26:~/scratch/mymodel". artistools starts "artistools server" on that host through ssh, and a
reader with the decorator on_model_host runs on that server. Only the result of the reader comes back,
thus the command copies no model file. The parquet caches stay beside the data on the remote host.

The pickle data comes only from the ssh process of the user, thus it is as safe as a command in ssh.
"""

import argparse
import functools
import re
import typing as t
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path

if t.TYPE_CHECKING:
    import subprocess
    import threading

    import polars as pl

REMOTEPATH_PATTERN = re.compile(r"^(?P<host>[A-Za-z0-9_][A-Za-z0-9_.@-]*):(?P<path>.*)$")

# a user can give a different command to start the server, e.g. the path of an artistools in a clone
SERVER_COMMAND_ENVVAR = "ARTISTOOLS_REMOTE_COMMAND"

# the server writes this line before its first message. A startup file of the remote shell can write text to the
# standard output first, thus the client ignores each line before this one
SERVER_START_LINE = b"artistools server protocol 1\n"

# each message is the length of its pickle data in this number of bytes, then the pickle data. The receiver reads
# a full message before it unpickles it, thus an error in the unpickle leaves the next message in step
MESSAGE_LENGTH_BYTES = 8


def split_remote_path(path: Path) -> tuple[str, Path] | None:
    """Return the host and the path on that host for a path of the form "host:path", or None for a local path.

    A local folder can have a colon in its name, thus a path that exists on this host stays local. ssh starts the
    server in the home folder, thus a relative path gets "~" at its start. The path is then absolute on the
    server, e.g. "vae26:" and "vae26:." name the home folder and not a folder with no name.
    """
    match = REMOTEPATH_PATTERN.match(str(path))
    if match is None or path.exists():
        return None

    hostpath = Path(match["path"])
    if not hostpath.is_absolute() and not match["path"].startswith("~"):
        hostpath = Path("~", hostpath)

    return match["host"], hostpath


def is_remote_path(path: Path | str) -> bool:
    """Return whether the path names a folder or a file on a different host."""
    return split_remote_path(Path(path)) is not None


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
    leaves out a part of the plot with no message.
    """
    if is_remote_path(path):
        msg = (
            f"artistools cannot read {path} with this option, because the file is on a different host."
            " Run the command on that host"
        )
        raise ValueError(msg)


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
    return ["ssh", host, os.environ.get(SERVER_COMMAND_ENVVAR) or defaultcommand]


def read_server_version(process: "subprocess.Popen[bytes]") -> str:
    """Return the version that the server sends after its start line. Raise EOFError if the server stops first."""
    import pickle

    assert process.stdout is not None
    while not (line := process.stdout.readline()).endswith(SERVER_START_LINE):
        if not line:
            raise EOFError

    serverversion = pickle.loads(read_message(process.stdout))
    assert isinstance(serverversion, str)
    return serverversion


@functools.cache
def get_server(host: str, pid: int) -> "tuple[subprocess.Popen[bytes], threading.Lock]":
    """Start the artistools server on the host, and return the process and the lock of its pipes.

    A child process cannot use the pipes of its parent, thus the process ID belongs to the key of the
    cache. The server stops when this process closes the pipes at its exit.
    """
    import os
    import shlex
    import subprocess  # ruff:ignore[suspicious-subprocess-import]
    import threading
    from importlib.metadata import version

    from artistools.misc.cliutils import exit_with_error
    from artistools.misc.cliutils import print_detail
    from artistools.misc.cliutils import print_warning

    assert pid == os.getpid()
    localversion = version("artistools")
    argv = get_server_argv(host)
    print_detail(f"artistools starts the server on {host} with: {shlex.join(argv)}")
    process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE)  # ruff:ignore[subprocess-without-shell-equals-true]

    try:
        serverversion = read_server_version(process)
    except Exception:  # ruff:ignore[blind-except]
        # text of a shell or a server of a different protocol can give any error of the unpickle
        process.kill()
        exit_with_error(
            f"the artistools server on {host} did not start. The ssh command line was: {shlex.join(argv)}",
            f"Install uv on {host}. The default command needs a release {localversion} of artistools with the server"
            f" command. As an alternative, set {SERVER_COMMAND_ENVVAR} to a command that starts such a server",
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
    import pickle

    process, lock = get_server(host, os.getpid())
    assert process.stdin is not None
    assert process.stdout is not None

    request = dump_message((modulename, qualname, args, kwargs))
    with lock:
        try:
            write_message(process.stdin, request)
            response = read_message(process.stdout)
        except BaseException as exc:
            # an exchange that stops in the middle, e.g. at Ctrl-C, leaves a part of a message in the pipes. The next
            # call would read it, thus the server stops, and the next call starts a new one
            process.kill()
            get_server.cache_clear()
            if isinstance(exc, (BrokenPipeError, EOFError)):
                msg = f"The artistools server on {host} stopped during a call of {qualname}"
                raise OSError(msg) from exc
            raise

    succeeded, result = pickle.loads(response)
    if not succeeded:
        assert isinstance(result, BaseException)
        result.add_note(f"The error came from {qualname} on the artistools server of {host}")
        raise result

    # the server gives back the paths of its own files, thus each one gets the name of the host
    return map_leaves(result, lambda leaf: Path(f"{host}:{leaf}") if isinstance(leaf, Path) else leaf)


def on_model_host[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Run the function on the host of the model when an argument is a remote path.

    A remote path is a Path object. A str argument stays local, because a label can have the form "name:text".
    The arguments go through pickle, thus they must be plain values, e.g. not an argparse.Namespace of the
    command. The result must be small. A LazyFrame comes back as the collected data. Put this decorator on a
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


def run_request(request: bytes) -> bytes:
    """Run one request of a client, and return the message with the result or the exception."""
    import pickle
    import traceback

    try:
        modulename, qualname, args, kwargs = pickle.loads(request)
        func = get_server_function(modulename, qualname)
        return dump_message((True, func(*map_leaves(args, expand_home), **map_leaves(kwargs, expand_home))))
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
