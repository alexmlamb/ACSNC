"""Static strain-based model."""

from math import sqrt, cos, sin, pi
import numpy as np
from scipy import interpolate, optimize

from .matrix_utils import big_adjoint, big_coadjoint, hat, exponential_map, little_coadjoint
from .kinematic import Kinematic


class Static(Kinematic):
    """The static strain-based model.

    Stores the configuration data for a model, and
    provides interface to solve for g based on
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
    ndof : dict
        dictionary between (tube_num, section_num) and degrees of freedom
    radius : dict
        dictionary of 'inner' and 'outer' radii, {str : list[float]}
    static_basis : StaticBasis
        class defining how to calculate the basis for the static equilibrium

    Attributes
    ----------
    ndof : dict
        dictionary between (tube_num, section_num) and degrees of freedom
    q_big_order : list[tuple]
        order of the independent basis q where each tuple is (tube_number, section_number)
    stiffness_matrices : list[float]
        the calculated stiffness matrix based off the given radius
    gauss_coefficients_kinematic : list[float]
    gauss_coefficients_static : list[float]
    gauss_weights : list[float]
    section_indices : dict
        the given indices for each section, to be reset by solve_g
    theta : list[float]
        the given thetas, to be reset by solve_g
    static_basis : StaticBasis
        class defining how to calculate the basis for the static equilibrium
    """
    def __init__(self, tube_num, q, q_dof, max_tube_length, delta_x, strain_base, ndof, radius, static_basis,
                 init_guess):
        super().__init__(tube_num, q, q_dof, max_tube_length, delta_x, strain_base)

        self.ndof = ndof
        self.q_big_order = [(1, 1), (2, 1), (3, 1), (2, 2), (3, 2), (3, 3)]  # how to use?
        self.stiffness_matrices = calculate_stiffness(radius)  # where to calculate?
        self.gauss_coefficients_kinematic, self.gauss_coefficients_static, self.gauss_weights = calculate_gauss()
        self.static_basis = static_basis
        self.section_indices = None
        self.theta = None
        self.general_init_guess = init_guess

    def solve_g(self, indices=None, thetas=None, initial_guess=None, full=False):
        if indices is None:  # default to zero insertion, with s(0) = L
            indices = [disc - 1 for disc in self.num_discrete_points]
        if thetas is None:
            thetas = [0] * self.tube_num

        if self.tube_num not in {3, 2}:
            raise ValueError(f'{self.tube_num} is not supported by the static model')

        # calculate indices at each section for each tube
        index_11 = indices[0]
        index_22 = indices[self.tube_num - 2]
        index_33 = indices[self.tube_num - 1]
        index_21 = index_22 - (self.num_discrete_points[0] - index_11 - 1)
        index_32 = index_33 - (self.num_discrete_points[self.tube_num - 2] - index_22 - 1)
        index_31 = index_32 - (self.num_discrete_points[0] - index_11 - 1)
        if self.tube_num == 2:
            index_11 = index_21 = index_31 = 0  # first section doesn't exist

        # store section indices in dictionary for convenience
        # {(tube_number, section_number) : index}
        section_indices = {(1, 1): index_11, (2, 1): index_21, (3, 1): index_31,
                           (2, 2): index_22, (3, 2): index_32, (3, 3): index_33}
        self.section_indices = section_indices
        self.theta = thetas  # type: list

        # if no initial_guess given, initialize array of zeros with 0.01 for each y, z constant bending - todo initial strain
        if initial_guess is None:
            initial_guess = self.general_init_guess

        # find root using MINPACKS hybrd and hybrj routines (modified Powell method)
        solution = optimize.root(self.static_equilibrium, x0=initial_guess, method='hybr')

        solution_q = solution.x
        print(solution.nfev)
        print(np.array2string(solution_q, separator=', '))
        print('\n')

        # first section
        if self.tube_num == 3:
            g_11 = self.integrate_g(solution_q, eye_from_theta(self.theta[0]), 1, 1)
            g_21 = self.integrate_g(solution_q, eye_from_theta(self.theta[1]), 2, 1)
            g_31 = self.integrate_g(solution_q, eye_from_theta(self.theta[2]), 3, 1)
        else:  # self.tube_num must equal 2
            # allows for easy access to initial SE(3) matrix for 2nd, 3rd sections
            g_11 = [np.eye(4)]
            g_21 = [eye_from_theta(self.theta[0])]
            g_31 = [eye_from_theta(self.theta[1])]

        # second section
        g_22 = self.integrate_g(solution_q, g_11[-1] @ g_21[-1], 2, 2)
        g_32 = self.integrate_g(solution_q, np.eye(4), 3, 2)

        # third section
        g_33 = self.integrate_g(solution_q, g_22[-1] @ g_31[-1] @ g_32[-1], 3, 3)  # todo check init

        if self.tube_num == 3:
            g_full = [g_11, g_22, g_33]
        else:
            g_full = [g_22, g_33]

        # reset attributes: section indices and theta for next run
        self.section_indices = None
        self.theta = None
        return g_full

    def solve_eta(self, velocity_list, prev_insert_indices_list, delta_theta_list, prev_g, curr_g):  # todo test
        ftl_out = []
        eta_out = []
        eta_previous_tube = np.zeros(6)
        for i in range(self.tube_num):
            current_length = len(curr_g[i])
            last_length = len(prev_g[i])
            index_range = range(max(0, current_length - last_length),
                                current_length)
            index_adjustment = last_length - current_length
            this_ftl_heuristic = []

            # todo how to calculate screw from curr_g/prev_g difference
            eta_here = [eta_previous_tube + curr_g[i][index_range[0]] - prev_g[i][index_range[0] + index_adjustment]]

            for x in index_range:
                g_prime = curr_g[i][x] - prev_g[i][x + index_adjustment]
                this_g = prev_g[i][x + index_adjustment]
                eta_r_local = big_adjoint(np.linalg.inv(this_g)) @ eta_here[0]
                ftl_here = eta_r_local - g_prime
                this_ftl_heuristic.append(ftl_here)

            ftl_out.append(this_ftl_heuristic)
            eta_out.append(eta_here)

        eta_out = [[np.zeros(6)]]
        return eta_out, ftl_out

    def integrate_g(self, solution, initial_g, tube_num, section_num):
        """Computes the g curve for this tube/section using the given solution q.

        Parameters
        ----------
        solution : np.ndarray
            The full solution q for all tubes and sections
        initial_g : np.ndarray
            The initial SE(3) g array for this tube/section
        tube_num : int
            This tube number: (2 or 3)
        section_num : int
            This section number: (1, 2, 3) if tube_num = 3; (2, 3) if tube_num = 2

        Returns
        -------
        list[np.ndarray]
            The integrated g_curve along the whole section
        """
        q = self.select_from_array(solution, tube_num, section_num)
        this_g = initial_g
        collect_g = [this_g]

        # gets desired section length (maps section 2 -> first item in num_discrete_points
        #  if there are only 2 tubes, and similar for other inputs)
        index_length_section = self.num_discrete_points[section_num + self.tube_num - 4] - self.section_indices.get(
            (section_num, section_num))
        start_index = self.section_indices.get((tube_num, section_num))

        for i in range(start_index + 1, start_index + index_length_section):
            _, this_g = self.half_step_integral(i, 0, q, this_g, tube_num, section_num)
            collect_g.append(this_g)

        return collect_g

    def select_from_array(self, arr, tube_num, section_num):
        """Helper to select correct sequence from the given array.

        The order of the array is assumed with self.q_big_order
        and each section is assumed to have the number of items
        given by self.ndof.

        Parameters
        ----------
        arr : np.ndarray
            The given array, with order specified in self.q_big_order
        tube_num : int
            The tube number, either 2 or 3
        section_num : int
            The section number (must be 2 or 3 if tube_num = 2)

        Returns
        -------
        np.ndarray
            The portion of arr specified by tube_num and section_num
        """
        end_ind = self.q_big_order.index((tube_num, section_num))
        flatten_dof = [self.ndof.get(item) for item in self.q_big_order]
        if end_ind == 0:
            return arr[0:flatten_dof[0]]
        else:
            return arr[sum(flatten_dof[0:end_ind]):sum(flatten_dof[0:end_ind + 1])]

    def static_equilibrium(self, q):
        """Provides the static equilibrium q based off the given q.

        To be used in a root optimization function.

        Parameters
        ----------
        q : np.ndarray
            Coordinates of the basis

        Returns
        -------
        np.ndarray
            Static equilibrium coordinates, for use in optimization

        Raises
        ------
        ValueError
            If the tube_num is not one of 2 or 3
        """

        if self.tube_num == 3:
            intc_1, intg_21, intg_31, sg31, gg31 = self.equilibrium_three(q)
            intg_312, intc_2, intg_32, intc_3 = self.equilibrium_two(q, sg31, gg31)
            return np.concatenate((intc_1, intg_21, intg_31 - intg_312, intc_2, intg_32, intc_3))

        elif self.tube_num == 2:  # only need second and third sections
            _, intc_2, intg_32, intc_3 = self.equilibrium_two(q, np.zeros((6, 2)), np.eye(4))
            return np.concatenate((intc_2, intg_32, intc_3))

        else:
            raise ValueError(f'{self.tube_num} number of tubes not supported by static model.')

    def equilibrium_three(self, q):
        """Helper equilibrium function for the first section using three tubes.

        Parameters
        ----------
        q : np.ndarray
            Coordinates of the basis

        Returns
        -------
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
            -- Section 1, tube 1 coordinates
            -- Section 1, tube 2 coordinates
            -- Section 1, tube 3 coordinates
            -- final sg for section 1, tube 3
            -- final gg (SE(3) g matrix) for section 1, tube 3
        """
        q_1 = [self.select_from_array(q, 1, 1),
               self.select_from_array(q, 2, 1),
               self.select_from_array(q, 3, 1)]

        ar_intc1 = []
        ar_intg21 = []
        ar_intg31 = []

        sg21 = np.zeros((6, self.ndof.get((2, 1))))
        sg31 = np.zeros((6, self.ndof.get((3, 1))))
        gg21 = eye_from_theta(self.theta[1])
        gg31 = eye_from_theta(self.theta[2])

        # iterate over length
        index_length_section_1 = self.num_discrete_points[0] - self.section_indices.get((1, 1))
        for i in range(index_length_section_1):
            index_adjusted = [i + self.section_indices.get((j, 1)) for j in range(1, 4)]

            bc11_here = self.static_basis.get_basis(i, 1, 1, self.section_indices, self.theta)
            bg21_here = self.static_basis.get_basis(i, 2, 1, self.section_indices, self.theta)
            bg31_here = self.static_basis.get_basis(i, 3, 1, self.section_indices, self.theta)

            ksi_1_here = bc11_here @ q_1[0] + self.strain_bias
            ksi_g_21_here = bg21_here @ q_1[1]
            ksi_31_here = bg31_here @ q_1[2]
            coadj_ksi_1_here = little_coadjoint(ksi_1_here)

            if i != 0:
                sg21, gg21 = self.half_step_integral(i, sg21, q_1[1], gg21, 2, 1)
                sg31, gg31 = self.half_step_integral(i, sg31, q_1[2], gg31, 3, 1)

            adj_g21_here = big_adjoint(gg21)
            adj_g31_here = big_adjoint(gg31)
            coadj_g21_here = big_coadjoint(gg21)
            coadj_g31_here = big_coadjoint(gg31)

            # diff
            ksi_21_here = np.linalg.inv(adj_g21_here) @ ksi_1_here + ksi_g_21_here
            ksi_31_here = np.linalg.inv(adj_g31_here) @ np.linalg.inv(
                adj_g21_here) @ ksi_1_here + np.linalg.inv(adj_g31_here) @ ksi_g_21_here + ksi_31_here

            fi_11_here = self.stiffness_matrices[0] @ (
                    ksi_1_here - self.get_ksi_star(index_adjusted[0], 1, self.q[0]))
            fi_21_here = self.stiffness_matrices[1] @ (
                    ksi_21_here - self.get_ksi_star(index_adjusted[1], 2, self.q[1]))
            fi_31_here = self.stiffness_matrices[2] @ (
                    ksi_31_here - self.get_ksi_star(index_adjusted[2], 3, self.q[2]))

            # diff
            ar_intc1.append(bc11_here.transpose() @ (
                    fi_11_here + coadj_g21_here @ fi_21_here +
                    coadj_g21_here @ coadj_g31_here @ fi_31_here))
            ar_intg21.append(bg21_here.transpose() @ (fi_21_here + fi_31_here) -
                             sg21.transpose() @ coadj_ksi_1_here @ coadj_g21_here @ (
                                     fi_21_here + coadj_g31_here @ fi_31_here))
            ar_intg31.append(bg31_here.transpose() @ fi_31_here -
                             sg31.transpose() @ coadj_ksi_1_here @
                             coadj_g21_here @ coadj_g31_here @ fi_31_here)

        # calculate integral
        intc_1 = self.calculate_integral(np.asarray(ar_intc1), index_length_section_1, self.ndof.get((1, 1)),
                                         self.gauss_coefficients_static, self.gauss_weights)
        intg_21 = self.calculate_integral(np.asarray(ar_intg21), index_length_section_1, self.ndof.get((2, 1)),
                                          self.gauss_coefficients_static, self.gauss_weights)
        intg_31 = self.calculate_integral(np.asarray(ar_intg31), index_length_section_1, self.ndof.get((3, 1)),
                                          self.gauss_coefficients_static, self.gauss_weights)

        return intc_1, intg_21, intg_31, sg31, gg31

    def equilibrium_two(self, q, sg31, gg31):
        """Helper equilibrium function for the first section using three tubes.

        Parameters
        ----------
        q : np.ndarray
            Coordinates of the basis
        sg31 : np.ndarray
            sg matrix for tube 3, last index of the first section
        gg31 : np.ndarray
            g matrix for tube 3, last index of the first section

        Returns
        -------
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
            -- Section 3, tube 1 coordinates, adjusted
            -- Section 2, tube 2 coordinates
            -- Section 2, tube 3 coordinates
            -- Section 3, tube 3 coordinates
        """
        # arranged as: [q_2, q_32]
        q_2 = [self.select_from_array(q, 2, 2),
               self.select_from_array(q, 3, 2)]

        ar_intc2 = []
        ar_intg32 = []
        ar_intg312 = []
        sg32 = np.zeros((6, self.ndof.get((3, 2))))
        sg312 = sg31

        # need 32: eye, 31: 31_previous
        if self.tube_num == 2:
            gg32 = eye_from_theta(self.theta[1])
        else:
            gg32 = np.eye(4)
        # gg31 = previous gg31
        adj_g31_here = big_adjoint(gg31)
        coadj_g31_here = big_coadjoint(gg31)

        # iterate over length
        index_length_section_2 = self.num_discrete_points[-2] - self.section_indices.get((2, 2))
        for i in range(index_length_section_2):
            index_adjusted = [i + self.section_indices.get((j, 2)) for j in range(2, 4)]

            # need 2, 32
            bc22_here = self.static_basis.get_basis(i, 2, 2, self.section_indices, self.theta)
            bg32_here = self.static_basis.get_basis(i, 3, 2, self.section_indices, self.theta)

            # need 2, 32
            ksi_2_here = bc22_here @ q_2[0] + self.strain_bias
            ksi_32_here = bg32_here @ q_2[1]
            coadj_ksi_2_here = little_coadjoint(ksi_2_here)

            if i != 0:
                sg32, gg32 = self.half_step_integral(i, sg32, q_2[1], gg32, 3, 2)

            # only 32
            adj_g32_here = big_adjoint(gg32)
            coadj_g32_here = big_coadjoint(gg32)

            # if only two tubes, adj_g31_here is identity
            ksi_32_here = np.linalg.inv(adj_g32_here) @ np.linalg.inv(adj_g31_here) @ ksi_2_here + ksi_32_here

            fi_22_here = self.stiffness_matrices[-2] @ (ksi_2_here -
                                                        self.get_ksi_star(index_adjusted[0],
                                                                          self.tube_num - 1, self.q[-2]))
            fi_32_here = self.stiffness_matrices[-1] @ (ksi_32_here -
                                                        self.get_ksi_star(index_adjusted[1],
                                                                          self.tube_num, self.q[-1]))

            # if only two tubes, coadj_g31_here is identity
            ar_intc2.append(bc22_here.transpose() @ (
                    fi_22_here + coadj_g31_here @ coadj_g32_here @ fi_32_here))
            ar_intg32.append(bg32_here.transpose() @ fi_32_here -
                             sg32.transpose() @ coadj_ksi_2_here @
                             coadj_g31_here @ coadj_g32_here @ fi_32_here)

            # no need for 312 if only two tubes, but leaving it in for simplicity
            ar_intg312.append(sg312.transpose() @ coadj_ksi_2_here @
                              coadj_g31_here @ coadj_g32_here @ fi_32_here)

        # calculate integral!
        # just x_vals and length different (and input integral values)
        intc_2 = self.calculate_integral(np.asarray(ar_intc2), index_length_section_2, self.ndof.get((2, 2)),
                                         self.gauss_coefficients_static, self.gauss_weights)
        intg_32 = self.calculate_integral(np.asarray(ar_intg32), index_length_section_2, self.ndof.get((3, 2)),
                                          self.gauss_coefficients_static, self.gauss_weights)
        intg_312 = self.calculate_integral(np.asarray(ar_intg312), index_length_section_2, self.ndof.get((3, 1)),
                                           self.gauss_coefficients_static, self.gauss_weights)

        # ---------------------- Section Three --------------------
        q_3 = self.select_from_array(q, 3, 3)

        ar_intc3 = []
        index_length_section_3 = self.num_discrete_points[self.tube_num - 1] - self.section_indices.get((3, 3))
        for i in range(index_length_section_3):
            index_adjusted = i + self.section_indices.get((3, 3))

            bc3_here = self.static_basis.get_basis(i, 3, 3, self.section_indices, self.theta)

            ksi_3_here = bc3_here @ q_3 + self.strain_bias
            fi_33_here = self.stiffness_matrices[self.tube_num - 1] @ (
                    ksi_3_here - self.get_ksi_star(index_adjusted, self.tube_num, self.q[-1]))
            ar_intc3.append(bc3_here.transpose() @ fi_33_here)
        intc_3 = self.calculate_integral(np.asarray(ar_intc3), index_length_section_3, self.ndof.get((3, 3)),
                                         self.gauss_coefficients_static, self.gauss_weights)

        return intg_312, intc_2, intg_32, intc_3

    def calculate_integral(self, ar_int, num_points, ndof, constants, weights):  # todo test
        """Computes gaussian quadrature of the given array.

        Parameters
        ----------
        ar_int : np.ndarray
            array values to be integrated
        num_points : int
            number of points in array
        ndof : int
            number of degrees of freedom for this section
        constants : list[float]
            gaussian constants
        weights : list[float]
            gaussian weights

        Returns
        -------
        np.ndarray
            The output integral with dof elements
        """
        length = (num_points - 1) * self.delta_x
        if length == 0:
            return np.zeros(ndof)
        x_vals = np.linspace(0, length, num_points)
        integral_collect = np.zeros((2, ndof))
        for i in range(ndof):  # todo allow for single (interp doesn't work)
            interp_function = interpolate.interp1d(x_vals, ar_int[:, i])
            this_iteration = map(lambda val: interp_function(val * length), constants)
            # this_iteration = [interp_function(val * length) for val in constants]
            integral_collect[:, i] = list(this_iteration)

        combine = map(lambda w, integ: w * integ, weights, integral_collect)
        combine2 = np.asarray(list(combine))
        iteration_out = length / 2 * combine2.sum(0)
        return iteration_out

    def half_step_integral(self, section_index, sg, this_q, gg, tube_number, section_number):
        """Integrate gg using two steps.

        Parameters
        ----------
        section_index
        sg : np.ndarray or int
            ???
        this_q : np.ndarray
            this set of coordinates
        gg : np.ndarray
            the previous g - SE(3) matrix
        tube_number : int
            this tube number (2 or 3)
        section_number : int
            this section number (if tube_num = 2, can be only 2 or 3)

        Returns
        -------
        (np.ndarray, np.ndarray)
            -- output sg matrix (6x1 matrix)
            -- output g matrix (4x4 SE(3))
        """
        step1 = section_index + self.gauss_coefficients_kinematic[0] - 1
        step2 = section_index + self.gauss_coefficients_kinematic[1] - 1
        bg_here1 = self.static_basis.get_basis(step1, tube_number, section_number, self.section_indices, self.theta)
        bg_here2 = self.static_basis.get_basis(step2, tube_number, section_number, self.section_indices, self.theta)
        sg = sg + (self.delta_x / 2) * (bg_here1 + bg_here2)

        if tube_number == section_number:
            ksig_hat_here1 = hat(bg_here1 @ this_q + self.strain_bias)
            ksig_hat_here2 = hat(bg_here2 @ this_q + self.strain_bias)
        else:  # only want torsion, no bias elongation
            ksig_hat_here1 = hat(bg_here1 @ this_q)
            ksig_hat_here2 = hat(bg_here2 @ this_q)

        # todo overkill - just an angle
        gammag_hat_here = (self.delta_x / 2) * (ksig_hat_here1 + ksig_hat_here2) + (
                (sqrt(3) * self.delta_x ** 2) / 12) * (
                                  ksig_hat_here2 @ ksig_hat_here1 - ksig_hat_here1 @ ksig_hat_here2)
        kg_here = np.transpose([-gammag_hat_here[1, 2], gammag_hat_here[0, 2], -gammag_hat_here[0, 1]])

        ggn_here = exponential_map(np.linalg.norm(kg_here), gammag_hat_here)
        gg = gg @ ggn_here

        return sg, gg

    def get_ksi_star(self, index, this_tube_num, q):
        """Get the initial strain of this tube with given q at given index.

        Parameters
        ----------
        index : int
            position along tube to calculate strain
        this_tube_num : int
            this tube number
        q : np.ndarray
            the given curvature values for this tube

        Returns
        -------
        np.ndarray
            the initial strain at this index
        """
        base_here = self.strain_base[this_tube_num - 1](index * self.delta_x, self.q_dof[this_tube_num - 1])
        return base_here @ q + self.strain_bias


def calculate_stiffness(radii):
    """Calculates the stiffness matrices for the given radii.

    Parameters
    ----------
    radii : dict
        lists defining the outer and inner radii for each tube;
        {'inner': list[float], 'outer': list[float]}

    Returns
    -------
    list[np.ndarray]
        stiffness matrix for each tube
    """
    r_out = radii.get('outer')
    r_in = radii.get('inner')
    area = [pi * (o ** 2 - i ** 2) for o, i in zip(r_out, r_in)]
    # Jy = Jz
    polar_mom = [pi / 4 * (o ** 4 - i ** 4) for o, i in zip(r_out, r_in)]
    mom_inertia = [pol * 2 for pol in polar_mom]
    e = 60e6  # 58e6 # kg mm^-1 s^-2
    g = 23.1e6  # 21.5e6

    return [np.diag(np.asarray([g * i, e * j, e * j, e * a, g * a, g * a])) for
            i, j, a in zip(mom_inertia, polar_mom, area)]


def calculate_gauss():  # todo use fewer order quadrature? could be an input
    """Collects the gauss quadrature coefficients in lists

    Returns
    -------
    (list[float], list[float], list[float])
        Gauss Quadrature coefficients for kinematics, for statics, and (static) weights
    """
    # Gauss quadrature coefficient for kinematics
    cg1 = 1 / 2 - sqrt(3) / 6
    cg2 = 1 / 2 + sqrt(3) / 6
    kinematic = [cg1, cg2]
    # Gauss quadrature coefficients for statics
    c1 = (1 / sqrt(3) + 1) / 2
    c2 = (-1 / sqrt(3) + 1) / 2
    static = [c1, c2]
    # c1 = 1 / 2 - (1 / 3) * sqrt(5 - 2 * sqrt(10 / 7)) / 2
    # c2 = 1 / 2 + (1 / 3) * sqrt(5 - 2 * sqrt(10 / 7)) / 2
    # c3 = 1 / 2 - (1 / 3) * sqrt(5 + 2 * sqrt(10 / 7)) / 2
    # c4 = 1 / 2 + (1 / 3) * sqrt(5 + 2 * sqrt(10 / 7)) / 2
    # static = [c1, c2, c3, c4, 0.5]
    # Gauss quadrature weights
    weights = [0.5, 0.5]
    # w1 = (322 + 13 * sqrt(70)) / 900
    # w3 = (322 - 13 * sqrt(70)) / 900
    # w5 = 128 / 225
    # weights = [w1, w1, w3, w3, w5]
    return kinematic, static, weights


def eye_from_theta(theta):
    """Calculates the rotation matrix theta around the x-axis

    Parameters
    ----------
    theta : float
        the given rotation

    Returns
    -------
    np.ndarray
        the rotation matrix
    """
    return np.asarray([[1, 0, 0, 0],
                       [0, cos(theta), -sin(theta), 0],
                       [0, sin(theta), cos(theta), 0],
                       [0, 0, 0, 1]])
