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
        self.time_slices = int(execution_window_sec / time_discretization_sec)
        self.shares_per_slice = total_shares // self.time_slices
        self.remaining_shares = total_shares % self.time_slices
        self.current_slice = 0
        self.time_discretization = time_discretization_sec


    def get_action(self, current_time_sec):
        """
        Get TWAP action for current time step

        Args:
            current_time_sec (float): Current time in the simulation in seconds

        Returns:
            int: Number of shares to execute in this time slice
        """

        slice_num = int(current_time_sec / self.time_discretization)

        if slice_num > self.current_slice:
            self.current_slice = slice_num

            # Calculate shares to execute
            if self.current_slice == self.num_slices - 1:
                # Last slice, so we execute the remaining shares
                return self.shares_per_slice + self.remaining_shares
            
            return self.shares_per_slice

        return 0 # no action needed this second


    def reset(self):
        """Reset the agent for a new episode"""
        self.current_slice = 0
        self.executed_shares = 0
