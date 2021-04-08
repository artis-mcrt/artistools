from pathlib import Path
import argparse
import artistools as at
import artistools.inputmodel.downscale3dgrid
import artistools.inputmodel.makeenergyinputfiles


def addargs(parser):
    parser.add_argument('-modelpath', default=[], nargs='*', action=at.AppendPath,
                        help='Path to initial model file')

    parser.add_argument('--downscale3dgrid', action='store_true',
                        help='Downscale a 3D ARTIS model to smaller grid size')

    parser.add_argument('--makeenergyinputfiles', action='store_true',
                        help='Downscale a 3D ARTIS model to smaller grid size')

    parser.add_argument('-modeldim', type=int, default=None,
                        help='Choose how many dimensions input model has. 1, 2 or 3')


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
        return

    if args.makeenergyinputfiles:
        model, t_model, vmax = at.inputmodel.get_modeldata(args.modelpath[0])
        if args.modeldim == 1:
            rho = 10**model['logrho']

        else:
            rho = model['rho']

        Mtot_grams = model['shellmass_grams'].sum()
        print(f"total mass { Mtot_grams / 1.989e33} Msun")

        at.inputmodel.makeenergyinputfiles.make_energy_files(rho, Mtot_grams)
        return


if __name__ == '__main__':
    main()
