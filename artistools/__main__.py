import argcomplete
import argparse
import importlib

import artistools.commands


def addargs(parser):
    pass


def main(argsraw=None):
    """Show a list of available artistools commands."""

    parser = argparse.ArgumentParser()
    parser.set_defaults(func=None)

    subparsers = parser.add_subparsers(dest='subcommand')
    subparsers.required = False

    for command, (submodulename, funcname) in sorted(artistools.commands.commandlist.items()):
        submodule = importlib.import_module(submodulename, package='artistools')
        subparser = subparsers.add_parser(command.replace('artistools-', ''))
        submodule.addargs(subparser)
        subparser.set_defaults(func=getattr(submodule, funcname))

    argcomplete.autocomplete(parser)
    args = parser.parse_args(argsraw)
    if args.func is not None:
        args.func(args=args)
    else:
        # parser.print_help()
        print('artistools provides the following commands:\n')

        # for script in sorted(console_scripts):
        #     command = script.split('=')[0].strip()
        #     print(f'  {command}')

        for command in sorted(artistools.commands.commandlist):
            print(f'  {command}')


if __name__ == '__main__':
    # multiprocessing.freeze_support()
    main()
