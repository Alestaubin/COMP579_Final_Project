# this is the code for the Time Weighted Average Price
import numpy as np

class TWAPAgent:
    def __init__(self, total_shares, total_time, time_discretization):
        """
        Initialize TWAP Agent

        Args:
            total_shares (int): Total number of shares to execute
            total_time (int): Total time available for execution (in time-steps)
            time_discretization (int): Time interval between orders (number of time steps) 
        """
        self.total_shares = total_shares
        self.total_time = total_time
        self.time_discretization = time_discretization

        # Calculate the number of slices of shares what will be traded
        self.num_slices = int(total_time / time_discretization)

        # Calculate the number of shares per slice, with the remainder of the shares being placed in the last slice
        self.shares_per_slice = total_shares // self.num_slices
        self.remaining_shares = total_shares % self.num_slices

        # fields to track the execution of the shares
        self.current_slice = 0
        self.executed_shares = 0


    def get_action(self, current_time):
        """
        Get TWAP action for current time step

        Args:
            current_time (float): Current time in the simulation

        Returns:
            int: Number of shares to execute in this time slice
        """

        if current_time >= self.total_time:
            return 0

        slice_num = int(current_time / self.time_discretization)

        if slice_num > self.current_slice:
            self.current_slice = slice_num

            # Calculate shares to execute
            if self.current_slice == self.num_slices - 1:
                # Last slice, so we execute the remaining shares
                return self.shares_per_slice + self.remaining_shares
            else:
                return self.shares_per_slice

        else:
            return 0


    def reset(self):
        """Reset the agent for a new episode"""
        self.current_slice = 0
        self.executed_shares = 0
