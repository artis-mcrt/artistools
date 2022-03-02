import matplotlib.pyplot as plt


def set_axis_properties(ax, args):
    if 'subplots' not in args:
        args.subplots = False
    if 'labelfontsize' not in args:
        args.labelfontsize = 18

    if args.subplots:
        for axis in ax:
            axis.minorticks_on()
            axis.tick_params(axis='both', which='minor', top=True, right=True, length=5, width=2,
                             labelsize=args.labelfontsize, direction='in')
            axis.tick_params(axis='both', which='major', top=True, right=True, length=8, width=2,
                             labelsize=args.labelfontsize, direction='in')

    else:
        ax.minorticks_on()
        ax.tick_params(axis='both', which='minor', top=True, right=True, length=5, width=2,
                       labelsize=args.labelfontsize, direction='in')
        ax.tick_params(axis='both', which='major', top=True, right=True, length=8, width=2,
                       labelsize=args.labelfontsize, direction='in')

    if 'ymin' in args or 'ymax' in args:
        plt.ylim(args.ymin, args.ymax)
    if 'xmin' in args or 'xmax' in args:
        plt.xlim(args.xmin, args.xmax)

    plt.minorticks_on()
    return ax


def set_axis_labels(fig, ax, xlabel, ylabel, labelfontsize, args):
    if args.subplots:
        fig.text(0.5, 0.02, xlabel, ha='center', va='center')
        fig.text(0.02, 0.5, ylabel, ha='center', va='center', rotation='vertical')
    else:
        ax.set_xlabel(xlabel, fontsize=labelfontsize)
        ax.set_ylabel(ylabel, fontsize=labelfontsize)
