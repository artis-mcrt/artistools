import unittest
from unittest.mock import MagicMock, patch, call
import argparse
import matplotlib.pyplot as plt  # Import for type hinting, will be mocked

import artistools.plottools as plottools

class TestPlottoolsRefactored(unittest.TestCase):
    def test_set_axis_properties_single_axis(self) -> None:
        mock_ax = MagicMock(spec=plt.Axes)
        args = argparse.Namespace(
            labelfontsize=12,
            ymin=0,
            ymax=10,
            xmin=1,
            xmax=20,
            logscalex=True,
            logscaley=False,
        )

        returned_ax = plottools.set_axis_properties(mock_ax, args)

        self.assertEqual(returned_ax, mock_ax)
        mock_ax.minorticks_on.assert_called_once()
        mock_ax.tick_params.assert_any_call(
            axis="both", which="minor", top=True, right=True, length=5, width=2, labelsize=12, direction="in"
        )
        mock_ax.tick_params.assert_any_call(
            axis="both", which="major", top=True, right=True, length=8, width=2, labelsize=12, direction="in"
        )
        mock_ax.set_ylim.assert_called_once_with(0, 10)
        mock_ax.set_xlim.assert_called_once_with(1, 20)
        mock_ax.set_xscale.assert_called_once_with("log")
        mock_ax.set_yscale.assert_not_called() # Or called with "linear" if that's the default path

    def test_set_axis_properties_iterable_axes(self) -> None:
        mock_ax1 = MagicMock(spec=plt.Axes)
        mock_ax2 = MagicMock(spec=plt.Axes)
        axes_list = [mock_ax1, mock_ax2]
        args = argparse.Namespace(
            labelfontsize=10,
            ymin=None, ymax=None, # Test case where limits are not set
            xmin=5, xmax=15,
            logscalex=False,
            logscaley=True,
        )

        returned_axes = plottools.set_axis_properties(axes_list, args)
        self.assertEqual(returned_axes, axes_list)

        for mock_ax in axes_list:
            mock_ax.minorticks_on.assert_called_once()
            mock_ax.tick_params.assert_any_call(
                axis="both", which="minor", top=True, right=True, length=5, width=2, labelsize=10, direction="in"
            )
            mock_ax.tick_params.assert_any_call(
                axis="both", which="major", top=True, right=True, length=8, width=2, labelsize=10, direction="in"
            )
            mock_ax.set_ylim.assert_not_called() # ymin/ymax are None
            mock_ax.set_xlim.assert_called_once_with(5, 15)
            mock_ax.set_xscale.assert_not_called()
            mock_ax.set_yscale.assert_called_once_with("log")

    def test_set_axis_properties_defaults(self) -> None:
        mock_ax = MagicMock(spec=plt.Axes)
        # Test with minimal args, relying on defaults set in the function
        args = argparse.Namespace()

        plottools.set_axis_properties(mock_ax, args)

        # Check that labelfontsize default is applied
        self.assertEqual(args.labelfontsize, 18)
        mock_ax.tick_params.assert_any_call(
            axis="both", which="major", top=True, right=True, length=8, width=2, labelsize=18, direction="in"
        )
        # Check that limits/scales are not called if args not present
        mock_ax.set_ylim.assert_not_called()
        mock_ax.set_xlim.assert_not_called()
        mock_ax.set_xscale.assert_not_called()
        mock_ax.set_yscale.assert_not_called()

if __name__ == "__main__":
    unittest.main()
