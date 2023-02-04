"""Follow the Leader module with various FTL classes."""

from .heuristic import Heuristic
import numpy as np


class FollowTheLeader(Heuristic):
    """Generic follow the leader class holding only the FTL magnitude.

    Parameters
    ----------
    only_tip : bool
        Is the FTL only calculated using the tip (rather than the
        average of the whole tube)?
    follow_the_leader : List[List[ndarray]]
        The NxLx6x1 ndarray defining the follow the leader screw for
        the curve (where N=tube_num, L==1 is only_tip=True)

    Attributes
    ----------
    ftl_magnitude : float
        The magnitude of this Heuristic's follow the leader array
    ftl_sum : float
        The sum follow the leader based on the parent's cost
    generation : int
        The generation of this Heuristic (+1 from its parent's)
    """

    def __init__(self, only_tip, follow_the_leader):
        if only_tip:
            ftl_tip_array = follow_the_leader[-1][-1]
            self.ftl_magnitude = self.calculate_magnitude(ftl_tip_array)  # type: float
        else:
            magnitudes = [self.calculate_magnitude(array) for tube in follow_the_leader for array in tube]
            self.ftl_magnitude = np.mean(magnitudes)  # type: float

        self.ftl_sum = self.ftl_magnitude
        self.generation = 0

    def calculate_cost_from_parent(self, parent: "FollowTheLeader", reset=False, init_insertion=False):
        if self.generation != 0 and not reset:
            print(f"Cost already calculated. Do not run method twice.")
            return
        self.ftl_sum = self._calculate_avg(parent.ftl_sum, parent.generation, self.ftl_magnitude)
        self.generation = parent.generation + 1

        if init_insertion:
            self.generation = 0
            self.ftl_sum = self.ftl_magnitude

    def test_cost_from_parent(self, parent: "FollowTheLeader"):
        return self._calculate_avg(parent.ftl_sum, parent.generation, self.ftl_magnitude)

    @staticmethod
    def _calculate_avg(parent_sum, parent_gen, this_ftl):
        """Calculate the average follow the leader cost.

        Uses the parent average and parent's generation (and the
        given FTL magnitude) to calculate the new average.

        Parameters
        ----------
        parent_sum : float
            The sum magnitude from the parent
        parent_gen : int
            The generation of the parent
        this_ftl : float
            This Heuristic's FTL magnitude

        Returns
        -------
        float
            The calculated average
        """
        return parent_sum + this_ftl

    def get_cost(self):
        return self.ftl_sum

    def get_own_cost(self):  # todo delete? - only used for debugging visually. Is there a need for this?
        """Gets the FTL magnitude (not the average magnitude along path).

        Returns
        -------
        float
        """
        return self.ftl_magnitude

    @staticmethod
    def calculate_magnitude(ftl_array):
        """Calculate the magnitude of the array given (of omega and velocity)

        Calculation is based on the magnitude of a screw, where the magnitude
        equals the Euclidean norm of the omega unless ||omega|| = 0 in which case
        the magnitude equals the Euclidean norm of the velocity.

        Parameters
        ----------
        ftl_array : list[float]
            omega and velocity values making up the follow-the-leader array,
            given as [O1, O2, O3, V1, V2, V3]

        Returns
        -------
        float
            magnitude of input array
        """
        omega_norm = np.linalg.norm(ftl_array[0:3])
        vel = ftl_array[3:6]

        # Magnitude of a screw equals the Euclidean norm of the omega
        #  or the norm of the velocity (if omega_norm = 0)
        if omega_norm == 0:
            return np.linalg.norm(vel)
        else:
            return omega_norm


class FollowTheLeaderWInsertion(FollowTheLeader):
    """Follow the leader class with insertion information.

    Parameters
    ----------
    only_tip : bool
        Is the FTL only calculated using the tip (rather than the
        average of the whole tube)?
    insertion_weight : float
        Weight of the insertion cost vs. the FTL cost
    follow_the_leader : List[List[ndarray]]
        The NxLx6x1 ndarray defining the follow the leader screw for
        the curve (where N=tube_num, L==1 is only_tip=True)
    insertion_fraction : float
        The decimal fraction [0, 1] defining how much the tubes
        are inserted out of the max insertion

    Attributes
    ----------
    ftl_magnitude : float
        The magnitude of this Heuristic's follow the leader array
    ftl_sum : float
        The sum of the follow the leader based on the parent's cost
    generation : int
        The generation of this Heuristic (+1 from its parent's)
    insertion_weight : float
        Weight of the insertion cost vs. the FTL cost
    insertion_fraction : float
        The decimal fraction [0, 1] defining how much the tubes
        are inserted out of the max insertion
        """

    def __init__(self, only_tip, insertion_weight, follow_the_leader, insertion_fraction):
        self.insertion_weight = insertion_weight
        self.insertion_fraction = insertion_fraction
        super().__init__(only_tip, follow_the_leader)

    def test_cost_from_parent(self, parent: "FollowTheLeaderWInsertion"):
        store_avg = self.ftl_sum
        self.ftl_sum = self._calculate_avg(parent.ftl_sum, parent.generation, self.ftl_magnitude)
        cost = self.get_cost()
        self.ftl_sum = store_avg
        return cost

    def get_cost(self):
        return self.ftl_sum + self.insertion_weight * self.insertion_fraction


class FollowTheLeaderTranslation(FollowTheLeader):

    def __init__(self, only_tip, follow_the_leader):
        super().__init__(only_tip, follow_the_leader)

    @staticmethod
    def calculate_magnitude(ftl_array):
        """Calculate the magnitude of the array given (of only velocity)

        Parameters
        ----------
        ftl_array : list[float]
            omega and velocity values making up the follow-the-leader array,
            given as [O1, O2, O3, V1, V2, V3]

        Returns
        -------
        float
            magnitude of last three values of array (translation velocity)
        """
        vel = ftl_array[3:6]

        return np.linalg.norm(vel)

