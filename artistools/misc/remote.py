# ruff:file-ignore[suspicious-pickle-import, suspicious-pickle-usage]
"""Run the readers of a model on the host that holds the model.

A model path of the form "host:path" names a folder on a host that ssh reaches, e.g.
"vae26:~/scratch/mymodel". The command starts "artistools server" on that host through ssh, and a
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

REMOTEPATH_PATTERN = re.compile(r"^(?P<host>[A-Za-z0-9_][A-Za-z0-9_.@-]*):(?P<path>.*)$")

# the command that starts the server on the remote host. A user can give a different command, e.g. the
# path of an artistools that is not on the PATH of a non-interactive ssh shell
SERVER_COMMAND_ENVVAR = "ARTISTOOLS_REMOTE_COMMAND"
DEFAULT_SERVER_COMMAND = "artistools server"


def split_remote_path(path: Path) -> tuple[str, Path] | None:
    """Return the host and the path on that host for a path of the form "host:path", or None for a local path.

    A local folder can have a colon in its name, thus a path that exists on this computer stays local.
    """
    match = REMOTEPATH_PATTERN.match(str(path))
    if match is None or path.exists():
        return None

    return match["host"], Path(match["path"] or ".")


def is_remote_path(path: Path | str) -> bool:
    """Return whether the path names a folder or a file on a different host."""
    return split_remote_path(Path(path)) is not None


def map_leaves(value: t.Any, func: Callable[[t.Any], t.Any]) -> t.Any:
    """Apply the function to each item inside the lists, the tuples, and the dict values of the value.

    Only these exact types hold the paths of an argument or a result, e.g. a list of model paths or a dict
    of LazyFrames for each direction bin. A NamedTuple or a different class stays as it is.
    """
    if type(value) is list:
        return [map_leaves(item, func) for item in value]
    if type(value) is tuple:
        return tuple(map_leaves(item, func) for item in value)
    if type(value) is dict:
        return {key: map_leaves(item, func) for key, item in value.items()}
    return func(value)


def find_remote_host(value: t.Any) -> str | None:
    """Return the host of the remote paths in the value, or None if all the paths are local.

    One call runs on one host, thus the paths of two hosts give an error.
    """
    hosts: set[str] = set()

    def add_host(leaf: t.Any) -> None:
        if isinstance(leaf, Path) and (remote := split_remote_path(leaf)) is not None:
            hosts.add(remote[0])

    map_leaves(value, add_host)
    if len(hosts) > 1:
        msg = f"One function cannot read the models of two hosts: {', '.join(sorted(hosts))}"
        raise ValueError(msg)

    return next(iter(hosts), None)


def to_server_path(leaf: t.Any) -> t.Any:
    """Return the path on the remote host for a path of the form "host:path"."""
    if isinstance(leaf, Path) and (remote := split_remote_path(leaf)) is not None:
        return remote[1]
    return leaf


@functools.cache
def get_server(host: str, pid: int) -> "tuple[subprocess.Popen[bytes], threading.Lock]":
    """Start the artistools server on the host, and return the process and the lock of its pipes.

    A child process cannot use the pipes of its parent, thus the process ID belongs to the key of the
    cache. The server stops when this process closes the pipes at its exit.
    """
    import os
    import pickle
    import subprocess  # ruff:ignore[suspicious-subprocess-import]
    import threading
    from importlib.metadata import version

    from artistools.misc.cliutils import print_detail
    from artistools.misc.cliutils import print_warning

    assert pid == os.getpid()
    servercommand = os.environ.get(SERVER_COMMAND_ENVVAR, DEFAULT_SERVER_COMMAND)
    print_detail(f"Starting the artistools server on {host} with: ssh {host} {servercommand}")
    # ssh gives the command to the shell of the remote host, thus the command is one string
    process = subprocess.Popen(  # ruff:ignore[subprocess-without-shell-equals-true]
        ["ssh", host, servercommand],  # ruff:ignore[start-process-with-partial-path]
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    assert process.stdout is not None

    try:
        serverversion = pickle.load(process.stdout)
    except (EOFError, pickle.UnpicklingError):
        process.kill()
        msg = (
            f"The artistools server on {host} did not start. The command was: ssh {host} {servercommand}. "
            f"Install artistools {version('artistools')} on {host}, or set {SERVER_COMMAND_ENVVAR} to the command "
            "that starts it"
        )
        raise OSError(msg) from None

    if serverversion != (localversion := version("artistools")):
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

    request = (modulename, qualname, map_leaves(args, to_server_path), map_leaves(kwargs, to_server_path))
    with lock:
        try:
            pickle.dump(request, process.stdin, protocol=pickle.HIGHEST_PROTOCOL)
            process.stdin.flush()
            succeeded, result = pickle.load(process.stdout)
        except (BrokenPipeError, EOFError) as exc:
            get_server.cache_clear()
            msg = f"The artistools server on {host} stopped during a call of {qualname}"
            raise OSError(msg) from exc

    if not succeeded:
        assert isinstance(result, BaseException)
        result.add_note(f"The error came from {qualname} on the artistools server of {host}")
        raise result

    # the server gives back the paths of its own files, thus each one gets the name of the host
    return map_leaves(result, lambda leaf: Path(f"{host}:{leaf}") if isinstance(leaf, Path) else leaf)


def on_model_host[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Run the function on the host of the model when an argument is a remote path.

    The server gives back the result of the function, thus the result must be small and must not refer to
    a file. A LazyFrame comes back as the collected data. Put this decorator on a function that reduces
    the data, e.g. a function that returns a spectrum, and not on a function that returns all the packets.
    """

    @functools.wraps(func)
    def run_on_model_host(*args: P.args, **kwargs: P.kwargs) -> R:
        host = find_remote_host((args, kwargs))
        if host is None:
            return func(*args, **kwargs)

        return t.cast("R", call_on_host(host, func.__module__, run_on_model_host.__qualname__, args, kwargs))

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


def collect_lazyframe(leaf: t.Any) -> t.Any:
    """Return the data of a LazyFrame as a LazyFrame that holds it in memory.

    The plan of a LazyFrame reads the files of the server, thus the client could not collect it.
    """
    import polars as pl

    return leaf.collect().lazy() if isinstance(leaf, pl.LazyFrame) else leaf


def run_request(request: tuple[str, str, tuple[t.Any, ...], dict[str, t.Any]]) -> tuple[bool, t.Any]:
    """Run one request of a client, and return whether it succeeded and the result or the exception."""
    import pickle
    import traceback

    modulename, qualname, args, kwargs = request
    try:
        func = get_server_function(modulename, qualname)
        result = map_leaves(func(*map_leaves(args, expand_home), **map_leaves(kwargs, expand_home)), collect_lazyframe)
        pickle.dumps(result)
    except (Exception, SystemExit) as exc:  # ruff:ignore[blind-except]
        # the client raises the same exception, thus a caller that catches FileNotFoundError still works
        traceback.print_exc()
        try:
            pickle.dumps(exc)
        except Exception:  # ruff:ignore[blind-except]
            exc = RuntimeError(f"{type(exc).__name__}: {exc}")
        return False, exc

    return True, result


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add no arguments. A client starts the server, and the server reads its requests from the standard input."""


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Run the readers that a different artistools process sends through ssh. A model path of host:path starts it."""
    import os
    import pickle
    import sys
    from importlib.metadata import version

    from artistools.misc.cliutils import parse_cli_args

    parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    # the pickled results use the standard output, thus a message that a reader prints goes to the standard
    # error. The client shows that stream to the user
    resultstream = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    requeststream = sys.stdin.buffer

    pickle.dump(version("artistools"), resultstream, protocol=pickle.HIGHEST_PROTOCOL)
    resultstream.flush()

    while True:
        try:
            request = pickle.load(requeststream)
        except EOFError:
            return

        pickle.dump(run_request(request), resultstream, protocol=pickle.HIGHEST_PROTOCOL)
        resultstream.flush()
