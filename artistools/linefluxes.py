#!/usr/bin/env python3
"""Artistools - spectra related functions."""
import argparse
import json
import math
import multiprocessing
from collections import namedtuple
# from functools import lru_cache
from functools import partial
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
# from astropy import constants as const
from astropy import units as u

import artistools as at
import artistools.packets


def get_packets_with_emtype_onefile(emtypecolumn, lineindices, packetsfile):
    import gzip
    try:
        dfpackets = at.packets.readfile(packetsfile, type='TYPE_ESCAPE', escape_type='TYPE_RPKT')
    except gzip.BadGzipFile:
        print(f'Bad file: {packetsfile}')
        raise gzip.BadGzipFile

    return dfpackets.query(f'{emtypecolumn} in @lineindices', inplace=False).copy()


@at.diskcache(savezipped=True)
def get_packets_with_emtype(modelpath, emtypecolumn, lineindices, maxpacketfiles=None):
    packetsfiles = at.packets.get_packetsfilepaths(modelpath, maxpacketfiles=maxpacketfiles)
    nprocs_read = len(packetsfiles)
    assert nprocs_read > 0

    model, _, _ = at.inputmodel.get_modeldata(modelpath)
    # vmax = model.iloc[-1].velocity_outer * u.km / u.s
    processfile = partial(get_packets_with_emtype_onefile, emtypecolumn, lineindices)
    if at.num_processes > 1:
        print(f"Reading packets files with {at.num_processes} processes")
        with multiprocessing.Pool(processes=at.num_processes) as pool:
            arr_dfmatchingpackets = pool.map(processfile, packetsfiles)
            pool.close()
            pool.join()
            # pool.terminate()
    else:
        arr_dfmatchingpackets = [processfile(f) for f in packetsfiles]

    dfmatchingpackets = pd.concat(arr_dfmatchingpackets)

    return dfmatchingpackets, nprocs_read


def calculate_timebinned_packet_sum(dfpackets, timearrayplusend):
    binned = pd.cut(dfpackets['t_arrive_d'], timearrayplusend, labels=False, include_lowest=True)

    binnedenergysums = np.zeros_like(timearrayplusend[:-1], dtype=float)
    for binindex, e_rf_sum in dfpackets.groupby(binned)['e_rf'].sum().iteritems():
        binnedenergysums[int(binindex)] = e_rf_sum

    return binnedenergysums


def get_line_fluxes_from_packets(emtypecolumn, emfeatures, modelpath, maxpacketfiles=None, arr_tstart=None, arr_tend=None):
    if arr_tstart is None:
        arr_tstart = at.get_timestep_times_float(modelpath, loc='start')
    if arr_tend is None:
        arr_tend = at.get_timestep_times_float(modelpath, loc='end')

    arr_timedelta = np.array(arr_tend) - np.array(arr_tstart)
    arr_tmid = arr_tend = (np.array(arr_tstart) + np.array(arr_tend)) / 2.

    model, _, _ = at.inputmodel.get_modeldata(modelpath)
    # vmax = model.iloc[-1].velocity_outer * u.km / u.s
    # betafactor = math.sqrt(1 - (vmax / const.c).decompose().value ** 2)

    timearrayplusend = np.concatenate([arr_tstart, [arr_tend[-1]]])

    dictlcdata = {'time': arr_tmid}

    linelistindices_allfeatures = tuple([l for feature in emfeatures for l in feature.linelistindices])

    dfpackets, nprocs_read = get_packets_with_emtype(
            modelpath, emtypecolumn, linelistindices_allfeatures, maxpacketfiles=maxpacketfiles)

    for feature in emfeatures:
        # dictlcdata[feature.colname] = np.zeros_like(arr_tstart, dtype=float)

        dfpackets_selected = dfpackets.query(f'{emtypecolumn} in @feature.linelistindices', inplace=False)

        normfactor = 1. / nprocs_read
        # mpc_to_cm = 3.085677581491367e+24  # 1 megaparsec in cm
        # normfactor = 1. / 4 / math.pi / (mpc_to_cm ** 2) / nprocs_read

        energysumsreduced = calculate_timebinned_packet_sum(dfpackets_selected, timearrayplusend)
        # print(energysumsreduced, arr_timedelta)
        fluxdata = np.divide(energysumsreduced * normfactor, arr_timedelta * u.day.to('s'))
        dictlcdata[feature.colname] = fluxdata

    lcdata = pd.DataFrame(dictlcdata)
    return lcdata


def get_line_fluxes_from_pops(emtypecolumn, emfeatures, modelpath, arr_tstart=None, arr_tend=None):
    import artistools.nltepops
    if arr_tstart is None:
        arr_tstart = at.get_timestep_times_float(modelpath, loc='start')
    if arr_tend is None:
        arr_tend = at.get_timestep_times_float(modelpath, loc='end')

    arr_timedelta = np.array(arr_tend) - np.array(arr_tstart)
    arr_tmid = arr_tend = (np.array(arr_tstart) + np.array(arr_tend)) / 2.

    modeldata, _, _ = at.inputmodel.get_modeldata(modelpath)

    ionlist = []
    for feature in emfeatures:
        ionlist.append((feature.atomic_number, feature.ion_stage))

    adata = at.atomic.get_levels(modelpath, ionlist=tuple(ionlist), get_transitions=True, get_photoionisations=False)

    timearrayplusend = np.concatenate([arr_tstart, [arr_tend[-1]]])

    dictlcdata = {'time': arr_tmid}

    for feature in emfeatures:
        fluxdata = np.zeros_like(arr_tmid, dtype=float)

        dfnltepops = at.nltepops.read_files(
            modelpath,
            dfquery=f'Z=={feature.atomic_number:.0f} and ion_stage=={feature.ion_stage:.0f}').query('level in @feature.upperlevelindicies')

        ion = adata.query(
            'Z == @feature.atomic_number and ion_stage == @feature.ion_stage').iloc[0]

        for timeindex, timedays in enumerate(arr_tmid):
            v_inner = modeldata.velocity_inner.values * u.km / u.s
            v_outer = modeldata.velocity_outer.values * u.km / u.s

            t_sec = timedays * u.day
            shell_volumes = ((4 * math.pi / 3) * ((v_outer * t_sec) ** 3 - (v_inner * t_sec) ** 3)).to('cm3').value

            timestep = at.get_timestep_of_timedays(modelpath, timedays)
            print(f'{feature.approxlambda}A {timedays}d (ts {timestep})')

            for upperlevelindex, lowerlevelindex in zip(feature.upperlevelindicies, feature.lowerlevelindicies):
                unaccounted_shellvol = 0.  # account for the volume of empty shells
                unaccounted_shells = []
                for modelgridindex in modeldata.index:
                    try:
                        levelpop = dfnltepops.query(
                            'modelgridindex==@modelgridindex and timestep==@timestep and Z==@feature.atomic_number'
                            ' and ion_stage==@feature.ion_stage and level==@upperlevelindex').iloc[0].n_NLTE

                        A_val = ion.transitions.query(
                                'upper == @upperlevelindex and lower == @lowerlevelindex').iloc[0].A

                        delta_ergs = (
                            ion.levels.iloc[upperlevelindex].energy_ev -
                            ion.levels.iloc[lowerlevelindex].energy_ev) * u.eV.to('erg')

                        # l = delta_ergs * A_val * levelpop * (shell_volumes[modelgridindex] + unaccounted_shellvol)
                        # print(f'  {modelgridindex} outer_velocity {modeldata.velocity_outer.values[modelgridindex]}'
                        #       f' km/s shell_vol: {shell_volumes[modelgridindex] + unaccounted_shellvol} cm3'
                        #       f' n_level {levelpop} cm-3 shell_Lum {l} erg/s')

                        fluxdata[timeindex] += delta_ergs * A_val * levelpop * (
                            shell_volumes[modelgridindex] + unaccounted_shellvol)

                        unaccounted_shellvol = 0.

                    except IndexError:
                        unaccounted_shellvol += shell_volumes[modelgridindex]
                        unaccounted_shells.append(modelgridindex)
                if unaccounted_shells:
                    print(f'No data for cells {unaccounted_shells} (expected for empty cells)')
                assert len(unaccounted_shells) < len(modeldata.index)  # must be data for at least one shell

        dictlcdata[feature.colname] = fluxdata

    lcdata = pd.DataFrame(dictlcdata)
    return lcdata


def get_closelines(modelpath, atomic_number, ion_stage, approxlambda, lambdamin=-1, lambdamax=-1, lowerlevelindex=-1, upperlevelindex=-1):
    dflinelist = at.get_linelist(modelpath, returntype='dataframe')
    dflinelistclosematches = dflinelist.query('atomic_number == @atomic_number and ionstage == @ion_stage').copy()
    if lambdamin > 0:
        dflinelistclosematches.query('@lambdamin < lambda_angstroms', inplace=True)
    if lambdamax > 0:
        dflinelistclosematches.query('@lambdamax > lambda_angstroms', inplace=True)
    if lowerlevelindex >= 0:
        dflinelistclosematches.query('lowerlevelindex==@lowerlevelindex', inplace=True)
    if upperlevelindex >= 0:
        dflinelistclosematches.query('upperlevelindex==@upperlevelindex', inplace=True)
    # print(dflinelistclosematches)

    linelistindices = tuple(dflinelistclosematches.index.values)
    upperlevelindicies = tuple(dflinelistclosematches.upperlevelindex.values)
    lowerlevelindicies = tuple(dflinelistclosematches.lowerlevelindex.values)
    lowestlambda = dflinelistclosematches.lambda_angstroms.min()
    highestlamba = dflinelistclosematches.lambda_angstroms.max()
    colname = f'flux_{at.get_ionstring(atomic_number, ion_stage, nospace=True)}_{approxlambda}'
    featurelabel = f'{at.get_ionstring(atomic_number, ion_stage)} {approxlambda} Å'

    return (colname, featurelabel, approxlambda, linelistindices, lowestlambda, highestlamba,
            atomic_number, ion_stage, upperlevelindicies, lowerlevelindicies)


def get_labelandlineindices(modelpath, emfeaturesearch):
    featuretuple = namedtuple('feature', [
        'colname', 'featurelabel', 'approxlambda', 'linelistindices', 'lowestlambda',
        'highestlamba', 'atomic_number', 'ion_stage', 'upperlevelindicies', 'lowerlevelindicies'])

    labelandlineindices = []
    for params in emfeaturesearch:
        feature = featuretuple(*get_closelines(modelpath, *params))
        print(f'{feature.featurelabel} includes {len(feature.linelistindices)} lines '
              f'[{feature.lowestlambda:.1f} Å, {feature.highestlamba:.1f} Å]')
        labelandlineindices.append(feature)
    # labelandlineindices.append(featuretuple(*get_closelines(dflinelist, 26, 2, 7155, 7150, 7160)))
    # labelandlineindices.append(featuretuple(*get_closelines(dflinelist, 26, 2, 12570, 12470, 12670)))
    # labelandlineindices.append(featuretuple(*get_closelines(dflinelist, 28, 2, 7378, 7373, 7383)))

    return labelandlineindices


def make_flux_ratio_plot(args):
    # font = {'size': 16}
    # matplotlib.rc('font', **font)
    nrows = 1
    fig, axes = plt.subplots(
        nrows=nrows, ncols=1, sharey=False,
        figsize=(args.figscale * at.figwidth, args.figscale * at.figwidth * (0.25 + nrows * 0.4)),
        tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    if nrows == 1:
        axes = [axes]

    axis = axes[0]
    axis.set_yscale('log')
    # axis.set_ylabel(r'log$_1$$_0$ F$_\lambda$ at 1 Mpc [erg/s/cm$^2$/$\mathrm{{\AA}}$]')

    # axis.set_xlim(left=supxmin, right=supxmax)
    pd.set_option('display.max_rows', 3500)
    pd.set_option('display.width', 150)
    pd.options.display.max_rows = 500

    for seriesindex, (modelpath, modellabel, modelcolor) in enumerate(zip(args.modelpath, args.label, args.color)):
        print(f"====> {modellabel}")

        emfeatures = get_labelandlineindices(modelpath, tuple(args.emfeaturesearch))

        if args.frompops:
            dflcdata = get_line_fluxes_from_pops(args.emtypecolumn, emfeatures, modelpath,
                                                 arr_tstart=args.timebins_tstart,
                                                 arr_tend=args.timebins_tend)
        else:
            dflcdata = get_line_fluxes_from_packets(args.emtypecolumn, emfeatures, modelpath,
                                                    maxpacketfiles=args.maxpacketfiles,
                                                    arr_tstart=args.timebins_tstart,
                                                    arr_tend=args.timebins_tend)

        dflcdata.eval(f'fratio = {emfeatures[1].colname} / {emfeatures[0].colname}', inplace=True)
        axis.set_ylabel(r'F$_{\mathrm{' + emfeatures[1].featurelabel + r'}}$ / F$_{\mathrm{' +
                        emfeatures[0].featurelabel + r'}}$')

        # \mathrm{\AA}

        print(dflcdata)

        axis.plot(dflcdata.time, dflcdata['fratio'], label=modellabel, marker='x', lw=0,
                  markersize=10, markeredgewidth=2,
                  color=modelcolor, alpha=0.8, fillstyle='none')

        tmin = dflcdata.time.min()
        tmax = dflcdata.time.max()

    if args.emfeaturesearch[0][:3] == (26, 2, 7155) and args.emfeaturesearch[1][:3] == (26, 2, 12570):
        axis.set_ylim(ymin=0.05)
        axis.set_ylim(ymax=7)
        arr_tdays = np.linspace(tmin, tmax, 3)
        arr_floersfit = [10 ** (0.0043 * timedays - 1.65) for timedays in arr_tdays]
        for ax in axes:
            ax.plot(arr_tdays, arr_floersfit, color='black', label='Flörs+2020 fit', lw=2.)

        femis = pd.read_csv(
            "/Users/luke/Dropbox/Papers (first-author)/2022 Artis ionisation/"
            "generateplots/floers_model_NIR_VIS_ratio_20201126.csv")

        amodels = {}
        for index, row in femis.iterrows():
            modelname = row.file.replace("fig-nne_Te_allcells-", "").replace(f"-{row.epoch}d.txt", "")
            if modelname not in amodels:
                amodels[modelname] = ([], [])
            if int(row.epoch) != 263:
                amodels[modelname][0].append(row.epoch)
                amodels[modelname][1].append(row.NIR_VIS_ratio)

        aindex = 0
        # for amodelname, (xlist, ylist) in amodels.items():
        for amodelname, alabel in [
                # ('w7', 'W7'),
                # ('subch', 'S0'),
                # ('subch_shen2018', r'1M$_\odot$'),
                # ('subch_shen2018_electronlossboost4x', '1M$_\odot$ (Shen+18) 4x e- loss'),
                # ('subch_shen2018_electronlossboost8x', r'1M$_\odot$ heatboost8'),
                # ('subch_shen2018_electronlossboost12x', '1M$_\odot$ (Shen+18) 12x e- loss'),
                ]:
            xlist, ylist = amodels[amodelname]
            color = args.color[aindex] if aindex < len(args.color) else None
            print(amodelname, xlist, ylist)
            axis.plot(xlist, ylist, color=color, label='Flörs ' + alabel, marker='+',
                      markersize=10, markeredgewidth=2, lw=0, alpha=0.8)
            aindex += 1

    m18_tdays = np.array([206, 229, 303, 339])
    m18_pew = {}
    # m18_pew[(26, 2, 12570)] = np.array([2383, 1941, 2798, 6770])
    m18_pew[(26, 2, 7155)] = np.array([618, 417, 406, 474])
    m18_pew[(28, 2, 7378)] = np.array([157, 256, 236, 309])
    if args.emfeaturesearch[1][:3] in m18_pew and args.emfeaturesearch[0][:3] in m18_pew:
        axis.set_ylim(ymax=12)
        arr_fratio = m18_pew[args.emfeaturesearch[1][:3]] / m18_pew[args.emfeaturesearch[0][:3]]
        for ax in axes:
            ax.plot(m18_tdays, arr_fratio, color='black', label='Maguire et al. (2018)', lw=2., marker='s')

    for ax in axes:
        ax.set_xlabel(r'Time [days]')
        if not args.nolegend:
            ax.legend(loc='upper right', frameon=False, handlelength=1, ncol=2, numpoints=1, prop={'size': 9})

    defaultoutputfile = 'linefluxes.pdf'
    if not args.outputfile:
        args.outputfile = defaultoutputfile
    elif not Path(args.outputfile).suffixes:
        args.outputfile = args.outputfile / defaultoutputfile

    fig.savefig(args.outputfile, format='pdf')
    # plt.show()
    print(f'Saved {args.outputfile}')
    plt.close()


@at.diskcache(savezipped=True)
def get_packets_with_emission_conditions(modelpath, emtypecolumn, lineindices, tstart, tend, maxpacketfiles=None):
    estimators = at.estimators.read_estimators(modelpath, get_ion_values=False, get_heatingcooling=False)

    modeldata, _, _ = at.inputmodel.get_modeldata(modelpath)
    ts = at.get_timestep_of_timedays(modelpath, tend)
    allnonemptymgilist = [modelgridindex for modelgridindex in modeldata.index
                          if not estimators[(ts, modelgridindex)]['emptycell']]

    # model_tmids = at.get_timestep_times_float(modelpath, loc='mid')
    # arr_velocity_mid = tuple(list([(float(v1) + float(v2)) * 0.5 for v1, v2 in zip(
    #     modeldata['velocity_inner'].values, modeldata['velocity_outer'].values)]))

    # from scipy.interpolate import interp1d
    # interp_log10nne, interp_te = {}, {}
    # for ts in range(len(model_tmids)):
    #     arr_v = np.zeros_like(allnonemptymgilist, dtype='float')
    #     arr_log10nne = np.zeros_like(allnonemptymgilist, dtype='float')
    #     arr_te = np.zeros_like(allnonemptymgilist, dtype='float')
    #     for i, mgi in enumerate(allnonemptymgilist):
    #         arr_v[i] = arr_velocity_mid[mgi]
    #         arr_log10nne[i] = math.log10(float(estimators[(ts, mgi)]['nne']))
    #         arr_te[i] = estimators[(ts, mgi)]['Te']
    #
    #     interp_log10nne[ts] =interp1d(arr_v.copy(), arr_log10nne.copy(),
    #                                                kind='linear', fill_value='extrapolate')
    #     interp_te[ts] = interp1d(arr_v.copy(), arr_te.copy(), kind='linear', fill_value='extrapolate')

    em_mgicolumn = 'em_modelgridindex' if emtypecolumn == 'emissiontype' else 'emtrue_modelgridindex'

    dfpackets_selected, _ = get_packets_with_emtype(
        modelpath, emtypecolumn, lineindices, maxpacketfiles=maxpacketfiles)

    dfpackets_selected = dfpackets_selected.query(
        't_arrive_d >= @tstart and t_arrive_d <= @tend', inplace=False).copy()

    dfpackets_selected = at.packets.add_derived_columns(dfpackets_selected, modelpath, ['em_timestep', em_mgicolumn],
                                                        allnonemptymgilist=allnonemptymgilist)

    if not dfpackets_selected.empty:
        def em_lognne(packet):
            # return interp_log10nne[packet.em_timestep](packet.true_emission_velocity)
            return math.log10(estimators[(int(packet['em_timestep']), int(packet[em_mgicolumn]))]['nne'])

        dfpackets_selected['em_log10nne'] = dfpackets_selected.apply(em_lognne, axis=1)

        def em_Te(packet):
            # return interp_te[packet.em_timestep](packet.true_emission_velocity)
            return estimators[(int(packet['em_timestep']), int(packet[em_mgicolumn]))]['Te']

        dfpackets_selected['em_Te'] = dfpackets_selected.apply(em_Te, axis=1)

    return dfpackets_selected


def plot_nne_te_points(axis, serieslabel, em_log10nne, em_Te, normtotalpackets, color, marker='o'):
    # color_adj = [(c + 0.3) / 1.3 for c in mpl.colors.to_rgb(color)]
    color_adj = [(c + 0.1) / 1.1 for c in mpl.colors.to_rgb(color)]
    hitcount = {}
    for log10nne, Te in zip(em_log10nne, em_Te):
        hitcount[(log10nne, Te)] = hitcount.get((log10nne, Te), 0) + 1

    if hitcount:
        arr_log10nne, arr_te = zip(*hitcount.keys())
    else:
        arr_log10nne, arr_te = np.array([]), np.array([])

    arr_weight = np.array([hitcount[(x, y)] for x, y in zip(arr_log10nne, arr_te)])
    arr_weight = (arr_weight / normtotalpackets) * 500
    arr_size = np.sqrt(arr_weight) * 10

    # arr_weight = arr_weight / float(max(arr_weight))
    # arr_color = np.zeros((len(arr_x), 4))
    # arr_color[:, :3] = np.array([[c for c in mpl.colors.to_rgb(color)] for x in arr_weight])
    # arr_color[:, 3] = (arr_weight + 0.2) / 1.2
    # np.array([[c * z for c in mpl.colors.to_rgb(color)] for z in arr_z])

    # axis.scatter(arr_log10nne, arr_te, s=arr_weight * 20, marker=marker, color=color_adj, lw=0, alpha=1.0,
    #              edgecolors='none')
    alpha = 0.8
    axis.scatter(arr_log10nne, arr_te, s=arr_size, marker=marker, color=color_adj, lw=0, alpha=alpha)

    # make an invisible plot series to appear in the legend with a fixed marker size
    axis.plot([0], [0], marker=marker, markersize=3, color=color_adj, linestyle='None', label=serieslabel, alpha=alpha)

    # axis.plot(em_log10nne, em_Te, label=serieslabel, linestyle='None',
    #           marker='o', markersize=2.5, markeredgewidth=0, alpha=0.05,
    #           fillstyle='full', color=color_b)


def plot_nne_te_bars(axis, serieslabel, em_log10nne, em_Te, color):
    if len(em_log10nne) == 0:
        return
    errorbarkwargs = dict(xerr=np.std(em_log10nne), yerr=np.std(em_Te),
                          color='black', markersize=10., fillstyle='full',
                          capthick=4, capsize=15, linewidth=4.,
                          alpha=1.0)
    # black larger one for an outline
    axis.errorbar(np.mean(em_log10nne), np.mean(em_Te), **errorbarkwargs)
    errorbarkwargs['markersize'] -= 2
    errorbarkwargs['capthick'] -= 2
    errorbarkwargs['capsize'] -= 1
    errorbarkwargs['linewidth'] -= 2
    errorbarkwargs['color'] = color
    # errorbarkwargs['zorder'] += 0.5
    axis.errorbar(np.mean(em_log10nne), np.mean(em_Te), **errorbarkwargs)


def make_emitting_regions_plot(args):
    import artistools.estimators

    # font = {'size': 16}
    # matplotlib.rc('font', **font)
    # 'floers_te_nne.json',
    refdatafilenames = ['floers_te_nne.json', ]  # , 'floers_te_nne_CMFGEN.json', 'floers_te_nne_Smyth.json']
    refdatalabels = ['Flörs+2020', ]  # , 'Floers CMFGEN', 'Floers Smyth']
    refdatacolors = ['0.0', 'C1', 'C2', 'C4']
    refdatakeys = [None for _ in refdatafilenames]
    refdatatimes = [None for _ in refdatafilenames]
    refdatapoints = [None for _ in refdatafilenames]
    for refdataindex, refdatafilename in enumerate(refdatafilenames):
        with open(refdatafilename, encoding='utf-8') as data_file:
            floers_te_nne = json.loads(data_file.read())

        # give an ordering and index to dict items
        refdatakeys[refdataindex] = [t for t in sorted(floers_te_nne.keys(), key=lambda x: float(x))]  # strings, not floats
        refdatatimes[refdataindex] = np.array([float(t) for t in refdatakeys[refdataindex]])
        refdatapoints[refdataindex] = [floers_te_nne[t] for t in refdatakeys[refdataindex]]
        print(f'{refdatafilename} data available for times: {list(refdatatimes[refdataindex])}')

    times_days = (np.array(args.timebins_tstart) + np.array(args.timebins_tend)) / 2.

    print(f'Chosen times: {times_days}')

    # axis.set_xlim(left=supxmin, right=supxmax)
    # pd.set_option('display.max_rows', 50)
    pd.set_option('display.width', 250)
    pd.options.display.max_rows = 500

    emdata_all = {}
    log10nnedata_all = {}
    Tedata_all = {}

    # data is collected, now make plots
    defaultoutputfile = 'emittingregions.pdf'
    if not args.outputfile:
        args.outputfile = defaultoutputfile
    elif not Path(args.outputfile).suffixes:
        args.outputfile = args.outputfile / defaultoutputfile

    args.modelpath.append(None)
    args.label.append(f'All models: {args.label}')
    args.modeltag.append('all')
    for modelindex, (modelpath, modellabel, modeltag) in enumerate(
            zip(args.modelpath, args.label, args.modeltag)):

        print(f"ARTIS model: '{modellabel}'")

        if modelpath is not None:
            print(f"Getting packets/nne/Te data for ARTIS model: '{modellabel}'")

            emdata_all[modelindex] = {}

            emfeatures = get_labelandlineindices(modelpath, tuple(args.emfeaturesearch))

            linelistindices_allfeatures = tuple([l for feature in emfeatures for l in feature.linelistindices])

            for tmid, tstart, tend in zip(times_days, args.timebins_tstart, args.timebins_tend):
                dfpackets = get_packets_with_emission_conditions(
                    modelpath, args.emtypecolumn, linelistindices_allfeatures, tstart, tend, maxpacketfiles=args.maxpacketfiles)

                for feature in emfeatures:
                    dfpackets_selected = dfpackets.query(f'{args.emtypecolumn} in @feature.linelistindices', inplace=False)
                    if dfpackets_selected.empty:
                        emdata_all[modelindex][(tmid, feature.colname)] = {
                            'em_log10nne': [],
                            'em_Te': []}
                    else:
                        emdata_all[modelindex][(tmid, feature.colname)] = {
                            'em_log10nne': dfpackets_selected.em_log10nne.values,
                            'em_Te': dfpackets_selected.em_Te.values}

            estimators = at.estimators.read_estimators(modelpath, get_ion_values=False, get_heatingcooling=False)
            modeldata, _, _ = at.inputmodel.get_modeldata(modelpath)
            Tedata_all[modelindex] = {}
            log10nnedata_all[modelindex] = {}
            for tmid, tstart, tend in zip(times_days, args.timebins_tstart, args.timebins_tend):
                Tedata_all[modelindex][tmid] = []
                log10nnedata_all[modelindex][tmid] = []
                tstartlist = at.get_timestep_times_float(modelpath, loc='start')
                tendlist = at.get_timestep_times_float(modelpath, loc='end')
                tslist = [ts for ts in range(len(tstartlist)) if tendlist[ts] >= tstart and tstartlist[ts] <= tend]
                for timestep in tslist:
                    for modelgridindex in modeldata.index:
                        try:
                            Tedata_all[modelindex][tmid].append(estimators[(timestep, modelgridindex)]['Te'])
                            log10nnedata_all[modelindex][tmid].append(math.log10(estimators[(timestep, modelgridindex)]['nne']))
                        except KeyError:
                            pass

        if modeltag != 'all':
            continue

        for tmid in times_days:
            print(f'  Plot at {tmid} days')

            nrows = 1
            fig, axis = plt.subplots(
                nrows=nrows, ncols=1, sharey=False, sharex=False,
                figsize=(args.figscale * at.figwidth, args.figscale * at.figwidth * (0.25 + nrows * 0.7)),
                tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.2})

            for refdataindex, f in enumerate(refdatafilenames):
                timeindex = np.abs(refdatatimes[refdataindex] - tmid).argmin()
                axis.plot(refdatapoints[refdataindex][timeindex]['ne'], refdatapoints[refdataindex][timeindex]['temp'],
                          color=refdatacolors[refdataindex], lw=2, label=f'{refdatalabels[refdataindex]} +{refdatakeys[refdataindex][timeindex]}d')

                timeindexb = np.abs(refdatatimes[refdataindex] - tmid - 50).argmin()
                if timeindexb < len(refdatakeys[refdataindex]):
                    axis.plot(refdatapoints[refdataindex][timeindexb]['ne'], refdatapoints[refdataindex][timeindexb]['temp'],
                              color='0.4', lw=2, label=f'{refdatalabels[refdataindex]} +{refdatakeys[refdataindex][timeindexb]}d')

            if modeltag == 'all':
                for bars in [False, ]:  # [False, True]
                    for truemodelindex in range(modelindex):
                        emfeatures = get_labelandlineindices(args.modelpath[truemodelindex], args.emfeaturesearch)

                        em_log10nne = np.concatenate(
                            [emdata_all[truemodelindex][(tmid, feature.colname)]['em_log10nne']
                             for feature in emfeatures])

                        em_Te = np.concatenate(
                            [emdata_all[truemodelindex][(tmid, feature.colname)]['em_Te']
                             for feature in emfeatures])

                        normtotalpackets = len(em_log10nne) * 8.  # circles have more area than triangles, so decrease
                        modelcolor = args.color[truemodelindex]
                        label = args.label[truemodelindex].format(timeavg=tmid, modeltag=modeltag)
                        if not bars:
                            plot_nne_te_points(
                                axis, label, em_log10nne, em_Te, normtotalpackets, modelcolor)
                        else:
                            plot_nne_te_bars(axis, args.label[truemodelindex], em_log10nne, em_Te, modelcolor)
            else:
                modellabel = args.label[modelindex]
                emfeatures = get_labelandlineindices(modelpath, tuple(args.emfeaturesearch))

                featurecolours = ['blue', 'red']
                markers = [10, 11]
                # featurecolours = ['C0', 'C3']
                # featurebarcolours = ['blue', 'red']

                normtotalpackets = np.sum([len(emdata_all[modelindex][(tmid, feature.colname)]['em_log10nne'])
                                           for feature in emfeatures])

                axis.scatter(log10nnedata_all[modelindex][tmid], Tedata_all[modelindex][tmid], s=1.0, marker='o',
                             color='0.4', lw=0, edgecolors='none', label='All cells')

                for bars in [False, ]:  # [False, True]
                    for featureindex, feature in enumerate(emfeatures):
                        emdata = emdata_all[modelindex][(tmid, feature.colname)]

                        if not bars:
                            print(f'   {len(emdata["em_log10nne"])} points plotted for {feature.featurelabel}')

                        serieslabel = (
                            modellabel + ' ' + feature.featurelabel).format(
                            timeavg=tmid, modeltag=modeltag).replace('Å', r' $\mathrm{\AA}$')

                        if not bars:
                            plot_nne_te_points(
                                axis, serieslabel, emdata['em_log10nne'], emdata['em_Te'],
                                normtotalpackets, featurecolours[featureindex], marker=markers[featureindex])
                        else:
                            plot_nne_te_bars(
                                axis, serieslabel, emdata['em_log10nne'], emdata['em_Te'], featurecolours[featureindex])

            if tmid == times_days[-1] and not args.nolegend:
                axis.legend(loc='best', frameon=False, handlelength=1, ncol=1, borderpad=0,
                            numpoints=1, fontsize=11, markerscale=2.5)

            axis.set_ylim(ymin=3000)
            axis.set_ylim(ymax=10000)
            axis.set_xlim(xmin=4.5, xmax=7.15)

            axis.set_xlabel(r'log$_{10}$(n$_{\mathrm{e}}$ [cm$^{-3}$])')
            axis.set_ylabel(r'Electron Temperature [K]')

            # axis.annotate(f'{tmid:.0f}d', xy=(0.98, 0.5), xycoords='axes fraction',
            #               horizontalalignment='right', verticalalignment='center', fontsize=16)

            outputfile = str(args.outputfile).format(timeavg=tmid, modeltag=modeltag)
            fig.savefig(outputfile, format='pdf')
            print(f'    Saved {outputfile}')
            plt.close()


def addargs(parser):
    parser.add_argument('-modelpath', default=[], nargs='*', action=at.AppendPath,
                        help='Paths to ARTIS folders with spec.out or packets files')

    parser.add_argument('-label', default=[], nargs='*',
                        help='List of series label overrides')

    parser.add_argument('--nolegend', action='store_true',
                        help='Suppress the legend from the plot')

    parser.add_argument('-modeltag', default=[], nargs='*',
                        help='List of model tags for file names')

    parser.add_argument('-color', default=[f'C{i}' for i in range(10)], nargs='*',
                        help='List of line colors')

    parser.add_argument('-linestyle', default=[], nargs='*',
                        help='List of line styles')

    parser.add_argument('-linewidth', default=[], nargs='*',
                        help='List of line widths')

    parser.add_argument('-dashes', default=[], nargs='*',
                        help='Dashes property of lines')

    parser.add_argument('-maxpacketfiles', type=int, default=None,
                        help='Limit the number of packet files read')

    parser.add_argument('-emfeaturesearch', default=[], nargs='*',
                        help='List of tuples (TODO explain)')

    # parser.add_argument('-emtypecolumn', default='trueemissiontype', choices=['emissiontype', 'trueemissiontype'],
    #                     help='Packet property for emission type - first thermal emission (trueemissiontype) '
    #                     'or last emission type (emissiontype)')

    parser.add_argument('--frompops', action='store_true',
                        help='Sum up internal emissivity instead of outgoing packets')

    parser.add_argument('--use_lastemissiontype', action='store_true',
                        help='Tag packets by their last scattering rather than thermal emission type')

    # parser.add_argument('-timemin', type=float,
    #                     help='Lower time in days to integrate spectrum')
    #
    # parser.add_argument('-timemax', type=float,
    #                     help='Upper time in days to integrate spectrum')
    #
    parser.add_argument('-xmin', type=int, default=50,
                        help='Plot range: minimum wavelength in Angstroms')

    parser.add_argument('-xmax', type=int, default=450,
                        help='Plot range: maximum wavelength in Angstroms')

    parser.add_argument('-ymin', type=float, default=None,
                        help='Plot range: y-axis')

    parser.add_argument('-ymax', type=float, default=None,
                        help='Plot range: y-axis')

    parser.add_argument('-timebins_tstart', default=[], nargs='*', action='append',
                        help='Time bin start values in days')

    parser.add_argument('-timebins_tend', default=[], nargs='*', action='append',
                        help='Time bin end values in days')

    parser.add_argument('-figscale', type=float, default=1.8,
                        help='Scale factor for plot area. 1.0 is for single-column')

    parser.add_argument('--write_data', action='store_true',
                        help='Save data used to generate the plot in a CSV file')

    parser.add_argument('--plotemittingregions', action='store_true',
                        help='Plot conditions where flux line is emitted')

    parser.add_argument('-outputfile', '-o', action='store', dest='outputfile', type=Path,
                        help='path/filename for PDF file')


def main(args=None, argsraw=None, **kwargs):
    """Plot spectra from ARTIS and reference data."""
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=at.CustomArgHelpFormatter,
            description='Plot ARTIS model spectra by finding spec.out files '
                        'in the current directory or subdirectories.')
        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    if not args.modelpath:
        args.modelpath = [Path('.')]
    elif isinstance(args.modelpath, (str, Path)):
        args.modelpath = [args.modelpath]

    args.modelpath = at.flatten_list(args.modelpath)

    args.label, args.modeltag, args.color = at.trim_or_pad(len(args.modelpath), args.label, args.modeltag, args.color)

    args.emtypecolumn = 'emissiontype' if args.use_lastemissiontype else 'trueemissiontype'

    for i in range(len(args.label)):
        if args.label[i] is None:
            args.label[i] = at.get_model_name(args.modelpath[i])

    if args.plotemittingregions:
        make_emitting_regions_plot(args)
    else:
        make_flux_ratio_plot(args)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
