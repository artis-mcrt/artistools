from pathlib import Path
import argparse
import artistools as at
import artistools.inputmodel.downscale3dgrid


def addargs(parser):
    parser.add_argument('-modelpath', default=[], nargs='*', action=at.AppendPath,
                        help='Path to initial model file')

    parser.add_argument('--downscale3dgrid', action='store_true',
                        help='Downscale a 3D ARTIS model to smaller grid size')


def main(args=None, argsraw=None, **kwargs):
    """Called with makeartismodel. Tools to create an ARTIS input model"""
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Make ARTIS input model')
        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    if not args.modelpath:
        args.modelpath = [Path('.')]
    elif isinstance(args.modelpath, (str, Path)):
        args.modelpath = [args.modelpath]

    args.modelpath = at.flatten_list(args.modelpath)

    if args.downscale3dgrid:
        at.inputmodel.downscale3dgrid.make_downscaled_3d_grid(modelpath=Path(args.modelpath[0]))


if __name__ == '__main__':
    main()
