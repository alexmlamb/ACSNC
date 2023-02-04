"""Class for Nelder Mead optimization."""

from scipy import optimize
import numpy as np
import time

from .optimizer import Optimizer
from ..model.model_factory import create_model
from ..solve.solver_factory import create_solver
from ctrdapp.model.strain_bases import max_from_base
from .optimize_result import OptimizeResult


class NelderMead(Optimizer):
    """Uses the nelder_mead optimize.minimize function to optimize the design.

    Attributes
    ----------
    solver_results : list[dict]
        Collects the results from each iteration (q and cost)
    best_solver : dict
        Contains the cost and Solver object of the best solver so far
    count : int
        The iteration count, to track progress
    """

    def __init__(self, heuristic_factory, collision_checker, initial_guess, configuration):
        super().__init__(heuristic_factory, collision_checker, initial_guess, configuration)

        self.solver_results = []
        self.best_solver = {'cost': 10e9, 'solver': None}
        self.count = 0

    def find_min(self):
        func = self.solver_heuristic

        radius = self.configuration.get('tube_radius').get('outer')
        if radius[0] == 5:  # large radius for single_tube ftl experiment
            radius = [0.9]

        init_simplex = calculate_simplex(self.configuration.get('strain_bases'), radius,
                                         self.configuration.get('tube_lengths'), self.configuration.get('q_dof'),
                                         self.initial_guess)

        # todo input initial simplex manually - sometimes algorithm needs finely tuned input paramaters
        # init_simplex = [[0.02, 0.02, -0.0005, 0.02, 0.0002, 0.01, 0.02, 0.0002, 0.01, -0.0002],
        #                 [0.02, 0.03, 0, 0.01, 0, 0.01, 0.02, 0, 0.02, 0],
        #                 [0.04, 0.02, -0.0004, 0.01, 0, 0, 0.03, -0.0003, 0.01, 0],
        #                 [0.01, 0.04, -0.0001, 0.03, -0.0002, 0.05, 0.02, 0.0003, 0, -0.0003],
        #                 [0, 0.04, -0.0008, 0, 0, 0, 0.02, 0.0003, 0, 0],
        #                 [0, 0.02, 0.0004, 0, 0, 0, 0.03, -0.0005, 0, 0],
        #                 [0, 0.03, -0.0008, 0.02, 0.0002, 0, 0, 0, 0.02, 0],
        #                 [0, 0.01, -0.0004, 0, 0, 0, 0.02, 0.0001, 0, 0.0003],
        #                 [0, 0.01, -0.0001, 0.01, 0.0002, 0, 0.01, 0.0002, 0.01, -0.001],
        #                 [0, 0.04, -0.0003, 0.04, -0.0001, 0, 0.04, 0.0002, 0.01, 0],
        #                 [0, 0.03, 0, 0, 0, 0, 0.005, 0, 0, 0]]

        print(init_simplex)

        # manual enter through command line
        # todo make this a GUI
        alter_simplex = input("Would you like to alter the simplex? (yes/no): ")
        while alter_simplex == 'yes' or alter_simplex == 'y':
            new_simplex = input("Write updated simplex (separate values by spaces, each iteration by semicolon (;): ")
            new_simplex = [[int(val) for val in s.strip().split(' ')] for s in new_simplex.split(';')]

            same_size = len(init_simplex) == len(new_simplex)
            if same_size:  # if iter num is same, check num of values in each iter
                for i, arr in enumerate(init_simplex):
                    same_size = same_size and len(arr) == len(new_simplex[i])
            if same_size:
                init_simplex = new_simplex
                alter_simplex = 'no'
            else:
                print('Input was not the correct size.')
                alter_simplex = input('Would you like to enter the simplex again? (yes/no): ')

        print(f'Using simplex:\n{init_simplex}')

        start_time = time.time()

        # todo allow for either initial guess or init_simplex
        optimize_result_nm = optimize.minimize(func, self.initial_guess,
                                               method='Nelder-Mead',
                                               options={'fatol': self.precision,
                                                        'xatol': self.precision,
                                                        'initial_simplex': init_simplex,
                                                        'maxfev': self.configuration.get('optimize_iterations')})
        end_time = time.time()

        optimize_result = OptimizeResult(optimize_result_nm.x,
                                         self.best_solver.get('solver'),
                                         optimize_result_nm.success,
                                         optimize_result_nm.nfev,
                                         optimize_result_nm.nit,
                                         end_time - start_time,
                                         self.solver_results)

        return optimize_result

    def solver_heuristic(self, x):
        """Calculates cost of solver with this given x.

        Also updates the solver_results and best_cost attributes.

        Parameters
        ----------
        x : list[float]
            input design component q

        Returns
        -------
        float
            cost of solver with given x
        """

        # model from x
        this_model = create_model(self.configuration, x)

        # model, heuristic_factory, collision_detector, configuration
        this_solver = create_solver(this_model, self.heuristic_factory, self.collision_checker, self.configuration)

        # call get_best_cost
        cost, index = this_solver.get_best_cost()

        solver_dict = {'q': this_solver.model.q, 'cost': cost}
        self.solver_results.append(solver_dict)

        if cost < self.best_solver.get('cost'):
            self.best_solver = {'cost': cost, 'solver': this_solver}

        self.count += 1
        print(f'{self.count} - Cost: {cost}, Solution: {this_solver.found_solution}, Q: {x}')

        return cost


# Q should be bounded (in both + and -) by (2*Emax)/Tube Diameter
# This is due to physical constraints on the bending curvature (nitinol
# must remain in the elastic/plastic region given by the Emax)
# The constant and changing q parameters must be considered to limit the
# max q possible
def calculate_simplex(base_names, tube_radii, length, q_dof, initial_guess, e_max=0.05):
    """Calculates an initial simplex to use in the Nelder Mead algorithm.

    If initial_guess is provided, bases simplex off of it. However, if any
    individual value is greater than the value given in the max_from_base
    function, the max value is used instead.

    Parameters
    ----------

    base_names : list[str]
        the base name for each tube
    tube_radii : list[float]
        the (outer) radius of each tube
    length : list[float]
        the length of each tube
    q_dof : list[int]
        the degrees of freedom for the base of each tube
    initial_guess : np.ndarray or None
        the initial guess to start the simplex from
    e_max : float
        the maximum strain allowed for a tube, (default is 0.05)

    Returns
    -------
    list[list[float]]
        The initial simplex
    """
    if initial_guess is None:
        initial_guess = np.ones(sum(q_dof)) * 100

    max_init_array = []
    for i in range(len(tube_radii)):
        tube_arr = max_from_base(base_names[i], e_max / tube_radii[i], length[i], q_dof[i])
        for ind, val in enumerate(tube_arr):
            init_index = sum(q_dof[0:i]) + ind
            if val > initial_guess[init_index]:
                max_init_array.append(initial_guess[init_index])
            else:
                max_init_array.append(val)

    dim = len(max_init_array)
    init_simplex = [[q / 2 for q in max_init_array]]
    for i in range(dim):
        new_array = max_init_array.copy()
        new_array[i] = new_array[i] / 4
        init_simplex.append(new_array)

    return init_simplex
