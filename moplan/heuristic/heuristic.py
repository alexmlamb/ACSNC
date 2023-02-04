"""Abstract heuristic class for calculating different cost metrics."""

from abc import ABC, abstractmethod


class Heuristic(ABC):
    """Abstract heuristic class; no abstract parameters."""

    @abstractmethod
    def calculate_cost_from_parent(self, parent, reset=False, init_insertion=False):
        """Calculates cost with the given heuristic as the parent.

        Calculates the cost with the given parent heuristic but does not
        return the cost (only mutates the Heuristic attributes). If reset
        is False, it assumes this is the first time the method has been
        called on this Heuristic instance. If reset is True, the cost is
        calculated anew without regards to the present data.

        Parameters
        ----------
        parent : Heuristic
            The heuristic of the parent node to this child node
        reset : bool
            Has this method been called before on this instance?
        init_insertion : bool
            Is this child node at an initial insertion state for all tubes?
            This designates a reset of the generations for it is possible
            to reach this node with pure rotation.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def test_cost_from_parent(self, parent):
        """Calculates theoretical cost using given heuristic as the parent.

        Calculates and returns the cost with the given parent heuristic
        but does not mutate the attributes.

        Parameters
        ----------
        parent : Heuristic
            The heuristic of the potential parent node to this child node

        Returns
        -------
        float
            The theoretical cost using given heuristic as the parent
        """
        pass

    @abstractmethod
    def get_cost(self):
        """Retrieve the cost of this heuristic.

        Returns
        -------
        float
        """
        pass
