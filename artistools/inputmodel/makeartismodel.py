from pathlib import Path
import argparse
import artistools as at
import artistools.inputmodel.downscale3dgrid
import artistools.inputmodel.energyinputfiles
import artistools.inputmodel.modelfromhydro
import artistools.inputmodel.opacityinputfile


def addargs(parser):
    parser.add_argument('-modelpath', default=[], nargs='*', action=at.AppendPath,
                        help='Path to initial model file')

    parser.add_argument('--downscale3dgrid', action='store_true',
                        help='Downscale a 3D ARTIS model to smaller grid size')

    parser.add_argument('-inputgridsize', default=200,
                        help='Size of big model grid for downscale script')

    parser.add_argument('-outputgridsize', default=50,
                        help='Size of small model grid for downscale script')

    parser.add_argument('--makemodelfromgriddata', action='store_true',
                        help='Make ARTIS model files from arepo grid.dat file')

    parser.add_argument('-pathtogriddata', default='.',
                        help='Path to arepo grid.dat file')

    parser.add_argument('--fillcentralhole', action='store_true',
                        help='Fill hole in middle of ejecta from arepo kilonova model')

    parser.add_argument('--getcellopacityfromYe', action='store_true',
                        help='Make opacity.txt where opacity is set in each cell by Ye from arepo model')

    parser.add_argument('--makeenergyinputfiles', action='store_true',
                        help='Downscale a 3D ARTIS model to smaller grid size')

    parser.add_argument('-modeldim', type=int, default=None,
                        help='Choose how many dimensions input model has. 1, 2 or 3')

    parser.add_argument('-outputpath', '-o', default='.',
                        help='Folder for output')


def main(args=None, argsraw=None, **kwargs):
    """Called with makeartismodel. Tools to create an ARTIS input model"""
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=at.CustomArgHelpFormatter,
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
        at.inputmodel.downscale3dgrid.make_downscaled_3d_grid(
            modelpath=Path(args.modelpath[0]), inputgridsize=args.inputgridsize, outputgridsize=args.outputgridsize)
        return

    if args.makemodelfromgriddata:
        print(args)
        dfmodel, t_model_days, vmax = at.inputmodel.modelfromhydro.read_griddat_file(args.pathtogriddata)

        if args.fillcentralhole:
            dfmodel = at.inputmodel.modelfromhydro.add_mass_to_center(dfmodel, t_model_days, vmax, args)

        if args.getcellopacityfromYe:
            at.inputmodel.opacityinputfile.opacity_by_Ye(args.modelpath[0], dfmodel)
        if 'cellYe' in dfmodel:
            at.inputmodel.opacityinputfile.write_Ye_file(args.modelpath[0], dfmodel)
        if 'Q' in dfmodel and args.makeenergyinputfiles:
            at.inputmodel.energyinputfiles.write_Q_energy_file(args.modelpath[0], dfmodel)

        if Path(args.pathtogriddata, 'gridcontributions.txt').is_file():
            print('found gridcontributions.txt. Adding abundances... (will be slow)')
            dfmodel, dfelabundances = at.inputmodel.rprocess_from_trajectory.add_abundancecontributions(
                gridcontribpath=args.pathtogriddata, dfmodel=dfmodel, t_model_days=t_model_days)
            print('Writing to abundances.txt...')
            at.inputmodel.save_initialabundances(dfelabundances=dfelabundances, abundancefilename=args.pathtogriddata)
        else:
            ngrid = len(dfmodel['rho'])
            at.inputmodel.save_empty_abundance_file(ngrid)

        print('Writing to model.txt...')
        at.inputmodel.save_modeldata(
            modelpath=args.modelpath[0], dfmodel=dfmodel, t_model_init_days=t_model_days, dimensions=3, vmax=vmax)

    if args.makeenergyinputfiles:
        model, t_model, vmax = at.inputmodel.get_modeldata(args.modelpath[0])
        if args.modeldim == 1:
            rho = 10**model['logrho']
            Mtot_grams = model['shellmass_grams'].sum()

        else:
            rho = model['rho']
            Mtot_grams = model['cellmass_grams'].sum()
        print(f"total mass { Mtot_grams / 1.989e33} Msun")

        at.inputmodel.energyinputfiles.make_energy_files(rho, Mtot_grams, outputpath=args.outputpath)


if __name__ == '__main__':
    main()
