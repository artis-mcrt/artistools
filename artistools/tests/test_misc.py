import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import argparse

import artistools.misc as misc

class TestMiscRefactored(unittest.TestCase):
    @patch("artistools.misc.get_nprocs")
    @patch("artistools.misc.get_npts_model")
    @patch("artistools.misc.get_mpirankofcell")
    def test_get_mpiranklist(
        self, mock_get_mpirankofcell: MagicMock, mock_get_npts_model: MagicMock, mock_get_nprocs: MagicMock
    ) -> None:
        mock_modelpath = Path("testmodel")
        mock_get_nprocs.return_value = 4
        mock_get_npts_model.return_value = 100

        # Test case 1: modelgridindex is None
        self.assertEqual(list(misc.get_mpiranklist(mock_modelpath, modelgridindex=None)), [0, 1, 2, 3])

        # Test case 2: modelgridindex is None, only_ranks_withgridcells=True
        mock_get_mpirankofcell.return_value = 2  # Assume last cell (99) is on rank 2
        self.assertEqual(
            list(misc.get_mpiranklist(mock_modelpath, modelgridindex=None, only_ranks_withgridcells=True)), [0, 1, 2]
        )
        mock_get_mpirankofcell.assert_called_with(modelpath=mock_modelpath, modelgridindex=99)

        # Test case 3: modelgridindex is an empty list
        self.assertEqual(list(misc.get_mpiranklist(mock_modelpath, modelgridindex=[])), [0, 1, 2, 3])

        # Test case 4: modelgridindex is a list with negative value
        self.assertEqual(list(misc.get_mpiranklist(mock_modelpath, modelgridindex=[10, -1, 20])), [0, 1, 2, 3])

        # Test case 5: modelgridindex is a single negative value
        self.assertEqual(list(misc.get_mpiranklist(mock_modelpath, modelgridindex=-5)), [0, 1, 2, 3])

        # Test case 6: modelgridindex is a list of valid cells
        mock_get_mpirankofcell.side_effect = lambda mgi, modelpath: mgi % 4
        self.assertEqual(list(misc.get_mpiranklist(mock_modelpath, modelgridindex=[0, 1, 5, 10])), [0, 1, 2])

        # Test case 7: modelgridindex is a single valid cell
        mock_get_mpirankofcell.side_effect = None # reset side_effect
        mock_get_mpirankofcell.return_value = 1
        self.assertEqual(list(misc.get_mpiranklist(mock_modelpath, modelgridindex=50)), [1])
        mock_get_mpirankofcell.assert_called_with(50, modelpath=mock_modelpath)

        # Test case 8: only_ranks_withgridcells=True and npts_model = 0
        mock_get_npts_model.return_value = 0
        self.assertEqual(
            list(misc.get_mpiranklist(mock_modelpath, modelgridindex=None, only_ranks_withgridcells=True)), []
        )
        mock_get_npts_model.return_value = 100 # reset for other tests


    @patch("artistools.misc.get_timestep_times")
    @patch("artistools.misc.get_model_name") # To avoid print warnings needing it
    @patch("artistools.misc.get_timestep_of_timedays")
    def test_get_time_range(self, mock_get_ts_of_day: MagicMock, mock_get_model_name:MagicMock, mock_get_ts_times: MagicMock) -> None:
        mock_modelpath = Path("testmodel")
        mock_get_model_name.return_value = "testmodel"

        # Mock get_timestep_times to return consistent values
        tstarts = [0.0, 1.0, 2.0, 3.0, 4.0]
        tmids = [0.5, 1.5, 2.5, 3.5, 4.5]
        tends = [1.0, 2.0, 3.0, 4.0, 5.0]
        mock_get_ts_times.side_effect = lambda modelpath, loc: {
            "start": tstarts, "mid": tmids, "end": tends
        }[loc]

        # Case 1: timestep_range_str
        ts_min, ts_max, td_low, td_up = misc.get_time_range(mock_modelpath, timestep_range_str="1-3")
        self.assertEqual((ts_min, ts_max), (1, 3))
        self.assertEqual((td_low, td_up), (tstarts[1], tends[3]))

        # Case 2: single timestep in timestep_range_str
        ts_min, ts_max, td_low, td_up = misc.get_time_range(mock_modelpath, timestep_range_str="2")
        self.assertEqual((ts_min, ts_max), (2, 2))
        self.assertEqual((td_low, td_up), (tstarts[2], tends[2]))

        # Case 3: timedays_range_str (range)
        ts_min, ts_max, td_low, td_up = misc.get_time_range(mock_modelpath, timedays_range_str="1.2-3.8", clamp_to_timesteps=True)
        self.assertEqual((ts_min, ts_max), (1, 3)) # tmid[1]=1.5 >= 1.2, tmid[3]=3.5 <= 3.8
        self.assertEqual((td_low, td_up), (tstarts[1], tends[3]))

        # Case 4: timedays_range_str (range), not clamped
        ts_min, ts_max, td_low, td_up = misc.get_time_range(mock_modelpath, timedays_range_str="1.2-3.8", clamp_to_timesteps=False)
        self.assertEqual((ts_min, ts_max), (1, 3))
        self.assertEqual((td_low, td_up), (1.2, 3.8))

        # Case 5: timedays_range_str (single value)
        mock_get_ts_of_day.return_value = 2 # timestep containing 2.5d is 2
        ts_min, ts_max, td_low, td_up = misc.get_time_range(mock_modelpath, timedays_range_str="2.5")
        self.assertEqual((ts_min, ts_max), (2, 2))
        self.assertEqual((td_low, td_up), (tstarts[2], tends[2])) # Clamped to timestep bounds

        # Case 6: timemin and timemax args
        ts_min, ts_max, td_low, td_up = misc.get_time_range(mock_modelpath, timemin=1.2, timemax=3.8, clamp_to_timesteps=True)
        self.assertEqual((ts_min, ts_max), (1, 3))
        self.assertEqual((td_low, td_up), (tstarts[1], tends[3]))

        # Case 7: timemin and timemax args, not clamped
        ts_min, ts_max, td_low, td_up = misc.get_time_range(mock_modelpath, timemin=1.2, timemax=3.8, clamp_to_timesteps=False)
        self.assertEqual((ts_min, ts_max), (1, 3))
        self.assertEqual((td_low, td_up), (1.2, 3.8))

        # Case 8: Edge case - timemin/max outside simulation range (should be caught by initial validation)
        # These are more like integration tests for the initial validation part.
        # We mock get_model_name because it's called in the warning print
        ts_min, ts_max, td_low, td_up = misc.get_time_range(mock_modelpath, timemin=10.0, timemax=12.0)
        self.assertEqual((ts_min, ts_max), (-1,-1)) # Expected error return
        self.assertEqual((td_low, td_up), (10.0, 12.0))

        ts_min, ts_max, td_low, td_up = misc.get_time_range(mock_modelpath, timemin=-2.0, timemax=-1.0)
        self.assertEqual((ts_min, ts_max), (-1,-1))
        self.assertEqual((td_low, td_up), (-2.0, -1.0))


        # Case 9: timestep_range_str out of bounds
        with self.assertRaises(ValueError):
            misc.get_time_range(mock_modelpath, timestep_range_str="10-12")
        with self.assertRaises(ValueError):
            misc.get_time_range(mock_modelpath, timestep_range_str="-1-2")
        with self.assertRaises(ValueError):
            misc.get_time_range(mock_modelpath, timestep_range_str="3-1") # max < min

        # Case 10: timedays leads to ts_max < ts_min (clamped)
        with self.assertRaises(ValueError):
             misc.get_time_range(mock_modelpath, timemin=3.6, timemax=3.4, clamp_to_timesteps=True) # tmids 3.5, 4.5. So ts_min=3, ts_max=2

        # Case 11: timedays leads to ts_max < ts_min (not clamped)
        # For days 3.6 to 3.4 (max < min):
        # input_td_min = 3.6, input_td_max = 3.4
        # ts_min becomes 3 (because tmid[3]=3.5 >= 3.6 is false, but tmid[4]=4.5 >=3.6 is true... wait, logic is tmid >= input_td_min. So tmid[3]=3.5 is NOT >= 3.6. tmid[4]=4.5 IS >= 3.6. So ts_min=4)
        # ts_max for 3.4: tmid[2]=2.5 <= 3.4. So ts_max=2.
        # This is ts_min=4, ts_max=2. The code sets ts_min = ts_max = 2.
        # td_low = 3.6, td_up = 3.4
        # This behavior (ts_min forced to ts_max) is one way to handle invalid range for non-clamped.
        ts_min, ts_max, td_low, td_up = misc.get_time_range(mock_modelpath, timemin=3.6, timemax=3.4, clamp_to_timesteps=False)
        self.assertEqual(td_low, 3.6)
        self.assertEqual(td_up, 3.4)
        self.assertEqual(ts_min, 2) # ts_min becomes ts_max
        self.assertEqual(ts_max, 2)


if __name__ == "__main__":
    unittest.main()
