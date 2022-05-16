import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


class ExponentLabelFormatter(ticker.ScalarFormatter):
    """Formatter to move the 'x10^x' offset text into the axis label."""

    def __init__(self, labeltemplate, useMathText=True, decimalplaces=None):
        self.set_labeltemplate(labeltemplate)
        self.decimalplaces = decimalplaces
        super().__init__(useOffset=True, useMathText=useMathText)
        # ticker.ScalarFormatter.__init__(self, useOffset=useOffset, useMathText=useMathText)

    def _set_formatted_label_text(self):
        # or use self.orderOfMagnitude
        stroffset = self.get_offset().replace(r'$\times', '$') + ' '
        strnewlabel = self.labeltemplate.format(stroffset)
        self.axis.set_label_text(strnewlabel)
        assert(self.offset == 0)
        self.axis.offsetText.set_visible(False)

    def set_labeltemplate(self, labeltemplate):
        assert '{' in labeltemplate
        self.labeltemplate = labeltemplate

    def set_locs(self, locs):
        if self.decimalplaces is not None:
            self.format = '%1.' + str(self.decimalplaces) + 'f'
            if self._usetex:
                self.format = '$%s$' % self.format
            elif self._useMathText:
                self.format = '$%s$' % ('\\mathdefault{%s}' % self.format)
        super().set_locs(locs)

        if self.decimalplaces is not None:
            # rounding the tick labels will make the locations incorrect unless we round these too
            newlocs = [float(('%1.' + str(self.decimalplaces) + 'f') % (x / (10 ** self.orderOfMagnitude)))
                       * (10 ** self.orderOfMagnitude) for x in self.locs]
            super().set_locs(newlocs)

        self._set_formatted_label_text()

    def set_axis(self, axis):
        super().set_axis(axis)
        self._set_formatted_label_text()


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


def imshow_init_for_artis_grid(ngrid, vmax, plot_variable_3d_array, plot_axes='xy'):
    # ngrid = round(len(model['inputcellid']) ** (1./3.))
    extent = {'left': -vmax, 'right': vmax, 'bottom': vmax, 'top': -vmax}
    extent = extent['left'], extent['right'], extent['bottom'], extent['top']
    data = np.zeros((ngrid, ngrid))

    plot_axes_choices = ['xy', 'zx']
    if plot_axes not in plot_axes_choices:
        print(f'Choose plot axes from {plot_axes_choices}')
        quit()

    for z in range(0, ngrid):
        for y in range(0, ngrid):
            for x in range(0, ngrid):
                if plot_axes == 'xy':
                    if z == round(ngrid/2)-1:
                        data[x, y] = plot_variable_3d_array[x, y, z]
                elif plot_axes == 'zx':
                    if y == round(ngrid/2)-1:
                        data[z, x] = plot_variable_3d_array[x, y, z]

    return data, extent


def autoscale(ax=None, axis='y', margin=0.1):
    '''Autoscales the x or y axis of a given matplotlib ax object
    to fit the margins set by manually limits of the other axis,
    with margins in fraction of the width of the plot

    Defaults to current axes object if not specified.
    From https://stackoverflow.com/questions/29461608/matplotlib-fixing-x-axis-scale-and-autoscale-y-axis
    '''

    def calculate_new_limit(fixed, dependent, limit):
        '''Calculates the min/max of the dependent axis given
        a fixed axis with limits
        '''
        if len(fixed) > 2:
            mask = (fixed > limit[0]) & (fixed < limit[1]) & (~np.isnan(dependent)) & (~np.isnan(fixed))
            window = dependent[mask]
            try:
                low, high = window.min(), window.max()
            except ValueError:  # Will throw ValueError if `window` has zero elements
                low, high = np.inf, -np.inf
        else:
            low = dependent[0]
            high = dependent[-1]
            if low == 0.0 and high == 1.0:
                # This is a axhline in the autoscale direction
                low = np.inf
                high = -np.inf
        return low, high

    def get_xy(artist):
        '''Gets the xy coordinates of a given artist
        '''
        if "Collection" in str(artist):
            x, y = artist.get_offsets().T
        elif "Line" in str(artist):
            x, y = artist.get_xdata(), artist.get_ydata()
        else:
            raise ValueError("This type of object isn't implemented yet")
        return x, y

    if ax is None:
        ax = plt.gca()
    newlow, newhigh = np.inf, -np.inf

    for artist in ax.collections + ax.lines:
        x, y = get_xy(artist)
        if axis == 'y':
            setlim = ax.set_ylim
            lim = ax.get_xlim()
            fixed, dependent = x, y
        else:
            setlim = ax.set_xlim
            lim = ax.get_ylim()
            fixed, dependent = y, x

        low, high = calculate_new_limit(fixed, dependent, lim)
        newlow = low if low < newlow else newlow
        newhigh = high if high > newhigh else newhigh

    margin = margin * (newhigh - newlow)

    setlim(newlow-margin, newhigh + margin)
