# this is the code for the Time Weighted Average Price
import numpy as np

class TWAPAgent:
    def __init__(self, total_shares, execution_window_sec, time_discretization_sec):
        """
        Initialize TWAP Agent

        Args:
            total_shares (int): Total number of shares to execute
            total_time (int): Total time available for execution (in time-steps)
            time_discretization (int): Time interval between orders (number of time steps) 
        """
        self.total_shares = total_shares
        self.time_slices = execution_window_sec // time_discretization_sec
        self.base_slice_shares = total_shares // self.time_slices
        self.remaining_shares = total_shares % self.time_slices
        self.slice_duration_pct = time_discretization_sec / execution_window_sec
        self.current_slice = -1


    def get_action(self, time_pct):
        """
        Get TWAP action for current time step

        Args:
            time_pct (float): percent of the total execution window we are at

        Returns:
            int: Number of shares to execute in this time slice
        """

        target_slice = int(time_pct / self.slice_duration_pct)
        if target_slice > self.current_slice:
            self.current_slice = target_slice
            if self.current_slice == self.time_slices - 1:
                return self.base_slice_shares + self.remaining_shares
            return self.base_slice_shares
        return 0


    def reset(self):
        """Reset the agent for a new episode"""
        self.current_slice = 0
        self.executed_shares = 0
