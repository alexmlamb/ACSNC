from abc import ABC, abstractmethod


class Model(ABC):
    """Models the mechanical system of the concentric tubes.

    The abstract class can be instantiated as any model (such as variable-strain
    kinematic and static). Model class primarily allows for calculating
    the g_curve for a given tube configuration and the eta.

    Parameters
    ----------
    tube_num : int
        number of tubes
    max_tube_length : list[float]
        maximum tube length for each tube
    delta_x : float
        size of this discretization

    Attributes
    ----------
    tube_num : int
        number of tubes
    max_tube_length : list[float]
        maximum tube length for each tube
    num_discrete_points : [int]
        number of discrete points along each tube,
        based on delta_x and max_tube_length
    delta_x : float
        size of this discretization
    """

    def __init__(self, tube_num, max_tube_length, delta_x):
        self.tube_num = tube_num
        self.max_tube_length = max_tube_length
        self.delta_x = delta_x
        self.num_discrete_points = [round(length / delta_x) + 1 for
                                    length in max_tube_length]

    @abstractmethod
    def solve_integrate(self, delta_theta, delta_insertion, this_theta, this_insertion, prev_g, invert_insert=True,
                        need_g_out=True):
        """Calculate the g and eta for one step in space.

        Parameters
        ----------
        need_g_out :
        delta_theta : list[float]
            change in theta values for each tube
        delta_insertion : list[float]
            change in insertion values for each tube
        this_theta : list[float]
            rotation values for each tube
        this_insertion : list[float]
            insertion values for each tube
        prev_g : list[list[np.ndarray]]
            4x4 SE3 g values for each tube from s to L of the previous insertion
        invert_insert : bool
            True if insertion values are given intuitively (with s(0) = 0 and
            s(L) = L; false otherwise (default is True))

        Returns
        -------
        (list[list[np.ndarray]], list[list[np.ndarray]], list[int], list[float], list[np.ndarray])
            --updated SE3 g values for each tube from s to L,
            where g_out[tube_num][index] = [4x4 SE3 array]
            --eta value for each tube, where
            eta_out[tube_num][0] = eta -> (eta stored in list for consistency)..
            --insertion indices for each tube, where
            insert_indices[tube_num] = int
            --true insertion values (rounded based on discretization),
            where true_insertions[tube_num] = float
            --follow-the-leader array (g_dot minus g_prime) for each tube from s to L,
            where ftl_out[tube_num] = [6x1 array]
        """
        pass

    @abstractmethod
    def solve_g(self, indices=None, thetas=None, **kwargs):
        """Calculates the g of each point for each tube at given index and theta

        At default, the tips of each tube will be aligned with the origin, with
        no rotation or insertion. If indices are given, the given index of each
        tube will be aligned with the tip of the previous tube with theta
        rotation along x-axis wrt previous tube.

        Parameters
        ---------
        indices : list[int]
            Insertion index for each tube, with initial as default (no insertion)
        thetas : list[float]
            Theta for each tube, with initial (0) as default
        kwargs
            Other optional parameters

        Returns
        -------
        list[list[np.ndarray]]
            g values for each tube,
            where g_out[tube number] = [4x4 SE3 array]
        """
        pass
