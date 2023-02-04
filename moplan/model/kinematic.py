"""Kinematic strain-based model."""

import numpy as np
from math import ceil, floor
import pathlib

from .model import Model
from .matrix_utils import big_adjoint, hat, exponential_map


class Kinematic(Model):
    """The kinematic strain-based model.

    Stores the configuration data for a model, and
    provides interface to solve for g and eta based on
    given configuration (s, theta, v, omega, etc.).

    Parameters
    ----------
    tube_num : int
        number of tubes
    q : np.ndarray[list[float]] or list[np.ndarray[float]]
        bending parameters for each tube
        (will be multiplied by the strain bases)
    q_dof : list[int]
        number of degrees of freedom for each tube
    max_tube_length : list[float]
        maximum tube length for each tube
    delta_x : float
        number of discrete points along each tube
    strain_base : list[function]
        function defining each tube's strain base

    Attributes
    ----------
    q : np.ndarray[list[float]] or list[np.ndarray[float]]
        bending parameters for each tube
        (will be multiplied by the strain bases)
    q_dof : list[int]
        number of degrees of freedom for each tube
    strain_base : list[function]
        function defining each tube's strain base
    strain_bias : np.ndarray
        bias strain for tube structure, x-elongation is default
    """
    def __init__(self, tube_num, q, q_dof, max_tube_length, delta_x, strain_base):
        super().__init__(tube_num, max_tube_length, delta_x)
        self.strain_base = strain_base
        self.strain_bias = np.array([0, 0, 0, 1, 0, 0])
        self.q_dof = q_dof
        self.q = q

    def solve_integrate(self, delta_theta, delta_insertion, this_theta, this_insertion, prev_g, invert_insert=True,
                        need_g_out=True):

        # kinematic model is defined as s(0)=L no insertion and s=0 for full insertion
        if invert_insert:
            this_insertion = [length - ins for ins, length in zip(this_insertion, self.max_tube_length)]
            delta_insertion = [-delta_ins for delta_ins in delta_insertion]

        prev_insertion = [ins - delta for ins, delta in zip(this_insertion, delta_insertion)]

        prev_insert_indices, new_insert_indices = calculate_indices(prev_insertion, this_insertion,
                                                                    self.max_tube_length, self.delta_x)
        if need_g_out:
            g_out = self.solve_g(indices=new_insert_indices, thetas=this_theta, full=False)
        else:
            g_out = [[]]

        # calculating velocity based on actual indices (modified for discretization)
        velocity = [(prev_i - new_i) * self.delta_x for new_i, prev_i in zip(new_insert_indices, prev_insert_indices)]
        eta_out, ftl_out = self.solve_eta(velocity, prev_insert_indices, delta_theta, prev_g, g_out)

        if invert_insert:
            true_insertions = [length - (ind * self.delta_x) for ind, length in
                               zip(new_insert_indices, self.max_tube_length)]
        else:
            true_insertions = [ind * self.delta_x for ind in new_insert_indices]

        return g_out, eta_out, new_insert_indices, true_insertions, ftl_out

    def solve_g(self, indices=None, thetas=None, full=True):
        if indices is None:  # default to zero insertion, with s(0) = L
            indices = [disc - 1 for disc in self.num_discrete_points]
        if thetas is None:
            thetas = [0] * self.tube_num

        g_out = []
        g_previous = np.eye(4)
        x_axis_unit = np.array([1, 0, 0, 0, 0, 0])

        for n in range(self.tube_num):
            # get this tube's initial g by applying theta rotation
            # to previous tube's final g
            theta_hat = hat(thetas[n] * x_axis_unit)
            theta_exp = exponential_map(thetas[n], theta_hat)
            g_previous = g_previous @ theta_exp
            this_g_tube = [g_previous]

            if full:
                index_start = 1
            else:
                index_start = indices[n] + 1

            # compute insertion w.r.t previous tube tip
            for i in range(index_start, self.num_discrete_points[n]):
                centered_s = (i - 0.5) * self.delta_x
                this_centered_base = self.strain_base[n](centered_s, self.q_dof[n])

                ksi_here_center = this_centered_base @ self.q[n] + self.strain_bias
                norm_k_here_center = np.linalg.norm(ksi_here_center[0:3])

                gn_here = exponential_map(self.delta_x * norm_k_here_center,
                                          self.delta_x * hat(ksi_here_center))
                g_here = g_previous @ gn_here

                this_g_tube.append(g_here)
                g_previous = g_here

            # move tube to align index with previous tip (if full tube is desired)
            # if tube is fully extended (index == 0), already aligned
            if full and indices[n] != 0:
                # transform full tube to align insert index with previous tube tip
                transform = this_g_tube[0] @ np.linalg.inv(this_g_tube[indices[n]])  # g_initial X g_index^-1
                this_g_tube = [transform @ g for g in this_g_tube]
                g_previous = this_g_tube[-1]

            g_out.append(this_g_tube)

        return g_out

    def solve_eta(self, velocity_list, prev_insert_indices_list, delta_theta_list, prev_g, curr_g):
        """Calculates eta and follow_the_leader array for given parameters.

        Parameters
        ----------
        curr_g : list[list[np.ndarray]]
            g values for each tube,
            where g_out[tube number] = [4x4 SE3 array]
        velocity_list : list[float]
            velocity (delta s) for each tube, previous - current(/new)
        prev_insert_indices_list : list[int]
            previous insert index for each tube
        delta_theta_list : list[float]
            delta theta for each tube, previous - current(/new)
        prev_g : list[list[np.ndarray]]
            4x4 SE3 g values for each tube from s to L of the previous insertion


        Returns
        -------
        (list[list[np.ndarray]], list[list[np.ndarray]])
            --eta value for each tube, where
            eta_out[tube_num][0] = eta -> (eta stored in list for consistency)..
            --follow-the-leader array (g_dot minus g_prime in local frame) for
            each tube from s to L, where ftl_out[tube_num] = [6x1 array]
        """
        eta_out = []
        ftl_out = []
        eta_previous_tube = np.zeros(6)  # in spatial frame
        x_axis_unit = np.array([1, 0, 0, 0, 0, 0])
        velocity_sum = 0

        for n in range(self.tube_num):
            # todo limit max delta_theta
            velocity_sum = velocity_sum + velocity_list[n]

            mid_point_x = prev_insert_indices_list[n] * self.delta_x - velocity_list[n] / 2

            ksi_here = self.strain_base[n](
                mid_point_x, self.q_dof[n]) @ self.q[n] + self.strain_bias

            # Adjoint of this tube's g_initial puts eta in spatial (world) frame
            eta_tr1 = big_adjoint(prev_g[n][0]) * delta_theta_list[n] @ x_axis_unit
            eta_tr2 = big_adjoint(prev_g[n][0]) * velocity_list[n] @ ksi_here

            eta_r_here = eta_previous_tube + eta_tr1 + eta_tr2
            this_eta_r = [eta_r_here]
            this_ftl_heuristic = []

            for i in range(prev_insert_indices_list[n], self.num_discrete_points[n]):
                this_prev_g_index = i - prev_insert_indices_list[n]

                # compute the g_prime based on the previous insertion
                this_base = self.strain_base[n](self.delta_x * i, self.q_dof[n])
                ksi_here = this_base @ self.q[n] + self.strain_bias

                # FTL calculated in the local frame
                g_prime = velocity_sum * ksi_here
                this_g = prev_g[n][this_prev_g_index]
                eta_r_local = big_adjoint(np.linalg.inv(this_g)) @ eta_r_here
                ftl_here = eta_r_local - g_prime
                this_ftl_heuristic.append(ftl_here)

            eta_previous_tube = eta_r_here
            eta_out.append(this_eta_r)  # eta is output w.r.t spatial frame
            ftl_out.append(this_ftl_heuristic)
        return eta_out, ftl_out


def calculate_indices(prev_insertions, next_insertions, max_tube_lengths, delta_x):
    """Calculates indices from given insertion values with respect to certain rules.

    Neither the previous nor the next insertion values can go beyond the range [0, max_tube_length]. The final delta
    must be an integer and is equal to 0 only if prev_insertion = next_insertion (the delta_index is always rounded
    up to 1 when the inherent delta is <1 and rounded to the nearest integer in all other cases). The previous insertion
    is always rounded to the nearest multiple of delta_x first and the next_insertion value is calculated from it and
    the rounded delta. All outputs are integers corresponding to indices (with the size of index span given by delta_x).

    Parameters
    ----------
    prev_insertions : list[float]
        previous insertion values
    next_insertions : list[float]
        next (desired) insertion values
    max_tube_lengths : list[float]
        maximum tube lengths, giving insertion bound for each tube
    delta_x : float
        The discretization size (size of one index)

    Returns
    -------
    (list[int], list[int])
        --the final index values for the previous insertion
        --the final index values for the next insertion
    """
    # todo check - should this change based on current insertion?
    length_sum = 0
    min_insertion = []
    for i in range(len(prev_insertions)):
        min_insertion.append(length_sum)
        length_sum = max_tube_lengths[i]

    prev_insert_indices = []
    new_insert_indices = []
    for n in range(len(prev_insertions)):
        max_length = max_tube_lengths[n]
        min_length = min_insertion[n]

        prev_insertion = prev_insertions[n]
        next_insertion = next_insertions[n]

        if prev_insertion < min_length:
            prev_insertion = min_length
        elif prev_insertion > max_length:
            prev_insertion = max_length

        if next_insertion < min_length:
            next_insertion = min_length
        elif next_insertion > max_length:
            next_insertion = max_length

        delta_index = next_insertion - prev_insertion

        if abs(delta_index) <= (0.5 * delta_x):
            delta_index = np.sign(delta_index)
            if round(prev_insertion / delta_x) == round(next_insertion / delta_x) and (
                    ceil(prev_insertion / delta_x) == ceil(next_insertion / delta_x) or
                    floor(prev_insertion / delta_x) == floor(next_insertion / delta_x)):
                if delta_index < 0:
                    prev_insert_index = int(ceil(prev_insertion / delta_x))
                else:
                    prev_insert_index = int(floor(prev_insertion / delta_x))
            else:
                prev_insert_index = int(round(prev_insertion / delta_x))
        else:
            delta_index = round(delta_index / delta_x)
            prev_insert_index = int(round(prev_insertion / delta_x))

        new_insert_index = prev_insert_index + int(delta_index)
        prev_insert_indices.append(prev_insert_index)
        new_insert_indices.append(new_insert_index)

    return prev_insert_indices, new_insert_indices


def save_g_positions(this_g, filename):
    """Saves only the position portion of the given g curve.

    Parameters
    ----------
    this_g : list[list[np.ndarray]]
        given 4x4 SE3 g values for each tube
    filename : str
        filename to use for saving

    Returns
    -------
    None
    """
    path = pathlib.Path().absolute()
    file = path / "outputs" / filename

    f = open(file, "w")

    for i, tube in enumerate(this_g):
        f.write(f"Tube number: {i}\n\n")
        for g in tube:
            f.write(f"{g[0, 3]}, {g[1, 3]}, {g[2, 3]}\n")

    f.close()
