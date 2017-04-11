#!/usr/bin/env python3
import argparse
import glob
import math
import os.path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from astropy import constants as const
from astropy import units as u

import artistools as at


def main():
    """
        Plot the radiation field estimators and the fitted radiation field
        based on the fitted field parameters (temperature and scale factor W
        for a diluted blackbody)
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Plot ARTIS radiation field.')
    parser.add_argument('-path', action='store', default='./',
                        help='Path to radfield_nnnn.out files')
    parser.add_argument('-listtimesteps', action='store_true', default=False,
                        help='Show the times at each timestep')
    parser.add_argument('-timestep', type=int, default=-1,
                        help='Timestep number to plot, or -1 for last')
    parser.add_argument('-timestepmax', type=int, default=-1,
                        help='Make plots for all timesteps up to this timestep')
    parser.add_argument('-modelgridindex', type=int, default=0,
                        help='Modelgridindex to plot')
    parser.add_argument('--nospec', action='store_true', default=False,
                        help='Don\'t plot the emergent specrum')
    parser.add_argument('-xmin', type=int, default=1000,
                        help='Plot range: minimum wavelength in Angstroms')
    parser.add_argument('-xmax', type=int, default=20000,
                        help='Plot range: maximum wavelength in Angstroms')
    parser.add_argument('--normalised', default=False, action='store_true',
                        help='Normalise the spectra to their peak values')
    parser.add_argument('-o', action='store', dest='outputfile',
                        default='plotradfield_cell{0:03d}_{1:03d}.pdf',
                        help='Filename for PDF file')
    args = parser.parse_args()

    if args.listtimesteps:
        at.showtimesteptimes('spec.out')
    else:
        radfield_files = (
            glob.glob('radfield_????.out', recursive=True) +
            glob.glob('*/radfield_????.out', recursive=True) +
            glob.glob('radfield-????.out', recursive=True) + glob.glob('radfield.out', recursive=True))

        if not radfield_files:
            print("No radfield files found")
            return
        else:
            radfielddata = at.radfield.read_files(radfield_files, args.modelgridindex)

        if not args.timestep or args.timestep < 0:
            timestepmin = max(radfielddata['timestep'])
        else:
            timestepmin = args.timestep

        if not args.timestepmax or args.timestepmax < 0:
            timestepmax = timestepmin + 1
        else:
            timestepmax = args.timestepmax

        specfilename = 'spec.out'

        if not os.path.isfile(specfilename):
            specfilename = '../example_run/spec.out'

        if not os.path.isfile(specfilename):
            print(f'Could not find {specfilename}')
            return

        for timestep in range(timestepmin, timestepmax):
            radfielddata_currenttimestep = radfielddata.query('timestep==@timestep')

            if len(radfielddata_currenttimestep) > 0:
                outputfile = args.outputfile.format(args.modelgridindex, timestep)
                make_plot(radfielddata_currenttimestep, specfilename, timestep, outputfile,
                          xmin=args.xmin, xmax=args.xmax, modelgridindex=args.modelgridindex, nospec=args.nospec,
                          normalised=args.normalised)
            else:
                print(f'No data for timestep {timestep:d}')


def make_plot(radfielddata, specfilename, timestep, outputfile, xmin, xmax, modelgridindex, nospec=False,
              normalised=False):
    """
        Draw the bin edges, fitted field, and emergent spectrum
    """
    time_days = at.get_timestep_time(specfilename, timestep)

    print(f'Plotting timestep {timestep:d} (t={time_days})')

    fig, axis = plt.subplots(1, 1, sharex=True, figsize=(8, 4),
                             tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    ymax1 = at.radfield.plot_field_estimators(axis, radfielddata)
    ymax2 = at.radfield.plot_fitted_field(axis, radfielddata, xmin, xmax)

    ymax = max(ymax1, ymax2)

    if len(radfielddata) < 400:
        binedges = [const.c.to('angstrom/s').value / radfielddata['nu_lower'].iloc[1]] + \
            list(const.c.to('angstrom/s').value / radfielddata['nu_upper'][1:])
        axis.vlines(binedges, ymin=0.0, ymax=ymax, linewidth=0.5,
                    color='red', label='', zorder=-1, alpha=0.4)
    if not nospec:
        if not normalised:
            modeldata, t_model_init = at.get_modeldata('model.txt')
            v_surface = modeldata.loc[int(radfielddata.modelgridindex.max())].velocity * u.km / u.s  # outer velocity
            r_surface = (327.773 * u.day * v_surface).to('km')
            r_observer = u.megaparsec.to('km')
            scale_factor = (r_observer / r_surface) ** 2 / (2 * math.pi)
            print(f'Scaling emergent spectrum flux at 1 Mpc to specific intensity '
                  f'at surface (v={v_surface:.3e}, r={r_surface:.3e})')
            plot_specout(axis, specfilename, timestep, scale_factor=scale_factor)  # peak_value=ymax)
        else:
            plot_specout(axis, specfilename, timestep, peak_value=ymax)

    axis.annotate(f'Timestep {timestep:d} (t={time_days})\nCell {modelgridindex:d}',
                  xy=(0.02, 0.96), xycoords='axes fraction',
                  horizontalalignment='left', verticalalignment='top', fontsize=8)

    axis.set_xlabel(r'Wavelength ($\AA$)')
    axis.set_ylabel(r'J$_\lambda$ [erg/s/cm$^2$/$\AA$]')
    axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=100))
    axis.set_xlim(xmin=xmin, xmax=xmax)
    axis.set_ylim(ymin=0.0, ymax=ymax)

    axis.legend(loc='best', handlelength=2,
                frameon=False, numpoints=1, prop={'size': 13})

    print(f'Saving to {outputfile:s}')
    fig.savefig(outputfile, format='pdf')
    plt.close()


def plot_specout(axis, specfilename, timestep, peak_value=None, scale_factor=None):
    """
        Plot the ARTIS spectrum
    """

    print(f"Plotting {specfilename}")

    dfspectrum = at.spectra.get_spectrum(specfilename, timestep)
    if scale_factor:
        dfspectrum['f_lambda'] = dfspectrum['f_lambda'] * scale_factor
    if peak_value:
        dfspectrum['f_lambda'] = dfspectrum['f_lambda'] / dfspectrum['f_lambda'].max() * peak_value

    dfspectrum.plot(x='lambda_angstroms', y='f_lambda', ax=axis, linewidth=1.5, color='black', alpha=0.7,
                    label='Emergent spectrum (normalised)')


if __name__ == "__main__":
    main()
