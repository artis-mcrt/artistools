import lzma
import os.path
import time
from functools import wraps
from pathlib import Path

import artistools.configuration
import artistools.misc


def diskcache(
    ignoreargs=[], ignorekwargs=[], saveonly=False, quiet=False, savezipped=False, funcdepends=None, funcversion=None
):
    import pickle
    import hashlib

    def printopt(*args, **kwargs):
        if not quiet:
            print(*args, **kwargs)

    @wraps(diskcache)
    def diskcacheinner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # save cached files in the folder of the first file/folder specified in the arguments
            modelpath = None
            for arg in [*args, *kwargs.values()]:
                if modelpath is None:
                    try:
                        if os.path.isfile(arg):
                            modelpath = Path(arg).parent

                    except TypeError:
                        pass

            for arg in [*args, *kwargs.values()]:
                if modelpath is None:
                    try:
                        if os.path.isdir(arg):
                            modelpath = arg
                    except TypeError:
                        pass

            if modelpath is None:
                modelpath = Path()  # use current folder

            cachefolder = Path(modelpath, "__artistoolscache__.nosync")

            if cachefolder.is_dir():
                try:
                    import xattr

                    xattr.setxattr(cachefolder, "com.dropbox.ignored", b"1")
                except OSError:
                    pass
                except ModuleNotFoundError:
                    pass

            namearghash = hashlib.sha1()
            namearghash.update(func.__module__.encode("utf-8"))
            namearghash.update(func.__qualname__.encode("utf-8"))

            namearghash.update(
                str(tuple(arg for argindex, arg in enumerate(args) if argindex not in ignoreargs)).encode("utf-8")
            )

            namearghash.update(str({k: v for k, v in kwargs.items() if k not in ignorekwargs}).encode("utf-8"))

            namearghash_strhex = namearghash.hexdigest()

            # make sure modifications to any file paths in the arguments will trigger an update
            argfilesmodifiedhash = hashlib.sha1()
            for arg in args:
                try:
                    if os.path.isfile(arg):
                        argfilesmodifiedhash.update(str(os.path.getmtime(arg)).encode("utf-8"))
                except TypeError:
                    pass
            argfilesmodifiedhash_strhex = "_filesmodifiedhash_" + argfilesmodifiedhash.hexdigest()

            filename_nogz = Path(cachefolder, f"cached-{func.__module__}.{func.__qualname__}-{namearghash_strhex}.tmp")
            filename_xz = filename_nogz.with_suffix(".tmp.xz")
            filename_gz = filename_nogz.with_suffix(".tmp.gz")

            execfunc = True
            saveresult = False
            functime = -1

            if (filename_nogz.exists() or filename_xz.exists() or filename_gz.exists()) and not saveonly:
                # found a candidate file, so load it
                filename = (
                    filename_nogz if filename_nogz.exists() else filename_gz if filename_gz.exists() else filename_xz
                )

                filesize = Path(filename).stat().st_size / 1024 / 1024

                try:
                    printopt(f"diskcache: Loading '{filename}' ({filesize:.1f} MiB)...")

                    with artistools.misc.zopen(filename, "rb") as f:
                        result, version_filein = pickle.load(f)

                    if version_filein == str_funcversion + argfilesmodifiedhash_strhex:
                        execfunc = False
                    elif (not funcversion) and (not version_filein.startswith("funcversion_")):
                        execfunc = False
                    # elif version_filein == sourcehash_strhex:
                    #     execfunc = False
                    else:
                        printopt(f"diskcache: Overwriting '{filename}' (function version mismatch or file modified)")

                except Exception as ex:
                    # ex = sys.exc_info()[0]
                    printopt(f"diskcache: Overwriting '{filename}' (Error: {ex})")
                    pass

            if execfunc:
                timestart = time.perf_counter()
                result = func(*args, **kwargs)
                functime = time.perf_counter() - timestart

            if functime > 1:
                # slow functions are worth saving to disk
                saveresult = True
            else:
                # check if we need to replace the gzipped or non-gzipped file with the correct one
                # if we so, need to save the new file even though functime is unknown since we read
                # from disk version instead of executing the function
                if savezipped and filename_nogz.exists():
                    saveresult = True
                elif not savezipped and filename_xz.exists():
                    saveresult = True

            if saveresult:
                # if the cache folder doesn't exist, create it
                if not cachefolder.is_dir():
                    cachefolder.mkdir(parents=True, exist_ok=True)
                try:
                    import xattr

                    xattr.setxattr(cachefolder, "com.dropbox.ignored", b"1")
                except OSError:
                    pass
                except ModuleNotFoundError:
                    pass

                if filename_nogz.exists():
                    filename_nogz.unlink()
                if filename_gz.exists():
                    filename_gz.unlink()
                if filename_xz.exists():
                    filename_xz.unlink()

                fopen, filename = (lzma.open, filename_xz) if savezipped else (open, filename_nogz)
                with fopen(filename, "wb") as f:
                    pickle.dump(
                        (result, str_funcversion + argfilesmodifiedhash_strhex), f, protocol=pickle.HIGHEST_PROTOCOL
                    )

                filesize = Path(filename).stat().st_size / 1024 / 1024
                printopt(f"diskcache: Saved '{filename}' ({filesize:.1f} MiB, functime {functime:.1f}s)")

            return result

        # sourcehash = hashlib.sha1()
        # sourcehash.update(inspect.getsource(func).encode('utf-8'))
        # if funcdepends:
        #     try:
        #         for f in funcdepends:
        #             sourcehash.update(inspect.getsource(f).encode('utf-8'))
        #     except TypeError:
        #         sourcehash.update(inspect.getsource(funcdepends).encode('utf-8'))
        #
        # sourcehash_strhex = sourcehash.hexdigest()
        str_funcversion = f"funcversion_{funcversion}" if funcversion else "funcversion_none"

        return wrapper if artistools.configuration.get_config()["enable_diskcache"] else func

    return diskcacheinner
