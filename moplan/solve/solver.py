from abc import ABC, abstractmethod


class Solver(ABC):
    """Obtains a solution to the given path planning problem.

    The abstract class can be instantiated as any planner (such as RRT and its
    derivatives, PRM, etc.). The public methods get_cost and get_path return the
    minimum cost found and the path (configurations from root to node)
    associated with that minimum cost. The methods vis_full_search and
    vis_best_solution allow 3D plotting of the path for the best solution
    (minimum cost) and the entire search space.
    """

    def __init__(self, model, heuristic_factory, collision_detector, configuration):
        """
        Parameters
        ----------
        model : Model
            Model object containing design info for tube deformation calculation
        heuristic_factory : HeuristicFactory
            Heuristic object containing general parameters and method to create
            new Heuristic object
        collision_detector : CollisionChecker
            Contains obstacles, goal, and methods for collision queries
        configuration : dict
            Dictionary storing configuration variables
        """

        self.model = model
        self.tube_num = configuration.get("tube_number")
        """int : Number of tubes used in this algorithm run"""
        self.tube_rad = configuration.get("tube_radius").get("outer")
        """List of float : Radii of the tubes"""
        self.heuristic_factory = heuristic_factory
        self.cd = collision_detector
        self.found_solution = False

    @abstractmethod
    def get_path(self, index):
        """Retrieves the data from the root to the given index.

        Parameters
        ----------
        index : int
            Index value for desired starting configuration node

        Returns
        -------
        list of list of 4x4 array : g values for each tube
        list of list of float : insertion values for each tube
        list of list of float : rotation values for each tube
        """
        pass

    @abstractmethod
    def get_best_cost(self):
        """Retrieves the minimum cost, with priority given to solutions.

        Returns
        -------
        float : cost of the optimal configuration node
        int : index of the optimal node
        """
        pass

    @abstractmethod
    def save_best_solution(self, output_dir):
        """Saves the path of the best solution (minimum cost) to a text file.

        Parameters
        ----------
        output_dir : pathlib.Path
            directory in which to save the file

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def save_tree(self, output_dir):
        """Saves the information of all nodes in the Solver to a text file.

        Parameters
        ----------
        output_dir : pathlib.Path
            directory in which to save the file

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def visualize_full_search(self, output_dir, tube_num=None, with_solution=True):
        """Visualizes the entire search, including obstacles, goal, and tubes

        Parameters
        ----------
        output_dir : Path
            A pathlib PosixPath giving the full path in which to save the output
        tube_num : int, optional
            Specifies which tube to visualize (default is all ->
             saves a separate file for each tube)
        with_solution : bool
            Should the solution path be plotted?

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def visualize_best_solution(self, objects_file, output_dir):
        """Visualizes best solution including obstacles, goal, and tubes.

        Results in a .pdf file being saved in the output_dir showing
        the final configuration of the best solution.

        Parameters
        ---------
        objects_file : Path
            A pathlib PosixPath giving the full path for obstacle/goal json file
        output_dir : Path
            A pathlib PosixPath giving the full path in which to save the output

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def visualize_best_solution_path(self, objects_file, output_dir):
        """Visualizes path of best solution including obstacles, goal, and tubes.

        Results in a mp4 file being saved in the specified output_dir
        showing an animation of the solution path.

        Parameters
        ---------
        objects_file : Path
            A pathlib PosixPath giving the full path for obstacle/goal json file
        output_dir : Path
            A pathlib PosixPath giving the full path in which to save the output

        Returns
        -------
        None
        """

    @abstractmethod
    def visualize_from_index_path(self, index, objects_file, output_dir, filename):
        """Visualizes path from root to index with obstacles, goal, and tubes

        Results in a mp4 file being saved in the specified output_dir
        showing an animation of the desired path.

        Parameters
        ----------
        index : int
            Index value for desired ending configuration node
        objects_file : Path
            A pathlib PosixPath giving the full path for obstacle/goal json file
        filename : str
            A string specifying what name to solve the output with
        output_dir : Path
            A pathlib PosixPath giving the full path in which to save the output

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def visualize_from_index(self, index, objects_file, output_dir, filename):
        """Visualizes given index with obstacles, goal, and tubes.

        Results in a .pdf file being saved in the output_dir showing
        the desired index configuration.

        Parameters
        ----------
        index : int
            Index value for desired ending configuration node
        objects_file : Path
            A pathlib PosixPath giving the full path for obstacle/goal json file
        filename : str
            A string specifying what name to solve the output with
        output_dir : Path
            A pathlib PosixPath giving the full path in which to save the output

        Returns
        -------
        None
        """
        pass
