from scipy import spatial
import pympnn


class DynamicTree:
    """Stores tree as multiple KDtrees to allow for fast nearest neighbor search

    The list of nodes are stored separately from the KDtrees, which contain only
    the insertion data for each node. There is an array that links the indices
    of the KDtree values to the node list. The multiple KDtrees allow for
    dynamic insertion of new nodes, while maintaining efficient nearest neighbor
    search. A NN query is O(log n * log n) and amortized insertion is
    O(log n * log n / n).

    This static-to-dynamic transformation structure is taken from  J. L. Bentley
    and J. B. Saxe*. Decomposable searching problems I: Static-to-dynamic
    transformation. J. Algorithms 1(4):301–358, 1980., as described in Jeff
    Erickson's (University of Illinois at Urbana-Champagne) lecture on advanced
    data structures.
    """

    def __init__(self, dim, ins, rot, init_heuristic, init_g_curves, max_tree_size, scaling):
        """
        Parameters
        ----------
        scaling :
        max_tree_size :
        dim : int
            number of dimensions represented in the node data
        ins : list[float]
            initial insertion values
        rot : list[float]
            initial rotation values
        init_heuristic : Heuristic
            Cost function and information for this node
        init_g_curves : list of list of 4x4 arrays
            List of SE3 g arrays for each tube
        """
        self.tube_num = dim

        init_insert_indices = [len(init_g_curves[0]) - 1] * self.tube_num
        first_node = Node(ins, rot, dim, init_heuristic, init_g_curves, init_insert_indices)
        topology = [1, 2] * dim  # 1 = linear, 2 = circular (rotation) for each tube
        full_scale = scaling * dim  # weights for insertion vs. rotation (could be passed in todo)
        self.kdtree = pympnn.KDTree_topologies(dim * 2, 16, topology, full_scale, max_tree_size + 2)
        """KDtree: utilizes the c++ mpnn library representing KD-trees with accurate topology."""

        first_point = ins + rot
        first_point[::2] = ins
        first_point[1::2] = rot
        self.kdtree.add_point(first_point)

        self.nodes = [first_node]
        """list of node: initialized with first node, filled by insert method"""
        self.at_goal = []  # use if not test (= true) to check for empty list
        """list of int: gives the indices of the successful nodes."""

    def nearest_neighbor(self, x):
        """Finds the nearest neighbor in the tree to the given point

        Parameters
        ----------
        x : list[float]
            the given point

        Returns
        -------
        (list[float]; list[float]; int; list[float])
            array giving the insertion values of the nearest neighbor
            array giving the rotation values of the nearest neighbor
            the index of the nearest neighbor (as stored in nodes array)
            array giving the insertion values of the neighbor's parent
        """

        min_dist, node_ind = self.kdtree.k_nearest_neighbor(x, 1)  # returns lists

        final_node = self.nodes[node_ind[0]]

        parent_ind = final_node.parent
        if parent_ind is not None:
            parent_node = self.nodes[parent_ind]
        else:  # first node has no parent
            parent_node = final_node
        return final_node.insertion, final_node.rotation, node_ind[0], parent_node.insertion

    def find_all_nearest_neighbor(self, x, max_distance):
        """Finds all the nearest neighbors to x within given distance (up to 16 neighbors).

        Parameters
        ----------
        x: list[float]
            given point
        max_distance: float
            maximum distance from x allowed

        Returns
        -------
        list[int]
            the indices of the nearest neighbors (as stored in nodes array)
        """

        # move this to c++ code ??? todo

        dist = [0]
        i = 0
        while dist[-1] <= max_distance and i < 3:
            dist, indices = self.kdtree.k_nearest_neighbor(x, 4 * 2**i)  # try 4, then 8, then 16
            i = i + 1

        for ind, this_distance in enumerate(dist):
            if this_distance > max_distance:
                final_index = ind - 1
                break
            else:
                final_index = ind

        if final_index == -1:  # should return at least one nearest_neighbor
            final_index = 0

        return indices[0:final_index+1]

    def insert(self, ins, rot, parent, heuristic, g_curves, insert_indices):
        """Inserts the new point into the tree with the given data and parent

        Parameters
        ----------
        ins : list[float]
            The insertion values for the point to be added to the tree
        rot : list[float]
            The rotation values for the point to be added to the tree
        parent : int
            The index of the parent node for this new node
        heuristic : Heuristic
            The cost structure and information for this new node
        g_curves : list[list[np.ndarray]]
            4x4 SE3 arrays for this node where g_curves[tube #][index] = 4x4 SE3
        insert_indices : list[int]
            corresponds to index for "origin" SE3 array for each g_curve tube

        Returns
        -------
        VOID
        """
        init_insertion = set(ins) == {0.0}
        heuristic.calculate_cost_from_parent(self.nodes[parent].heuristic, init_insertion=init_insertion)
        new_node = Node(ins, rot, self.tube_num, heuristic, g_curves, insert_indices, parent)
        self.nodes.append(new_node)
        this_node_index = len(self.nodes) - 1
        self.nodes[parent].children.append(this_node_index)

        this_point = ins + rot
        this_point[::2] = ins
        this_point[1::2] = rot
        self.kdtree.add_point(this_point)

    def no_cycle(self, parent_ind, child_ind):
        """Checks whether there is a cycle formed with these parent and child nodes (indices).

        Parameters
        ----------
        parent_ind : int
            index of the parent node
        child_ind : int
            index of the child node

        Returns
        -------
        Boolean
            True if no cycle exists; False if there is a cycle

        Raises
        ------
        ValueError
            If the given parent index is the root (0)
        """
        ancestor = self.nodes[parent_ind].parent

        if ancestor == child_ind:
            return False
        elif ancestor == 0:
            return True
        elif ancestor is None:
            raise ValueError('Parent_ind should never be entered as 0 (root node).')
        else:
            return self.no_cycle(ancestor, child_ind)

    def reset_heuristic_all_children(self, ind):
        """Reset the heuristic for all descendants of the given node (index)

        Parameters
        ----------
        ind : int
            Index of desired node

        Returns
        -------
        VOID
        """
        parent_heuristic = self.nodes[ind].heuristic
        children = self.nodes[ind].children
        for ch in children:
            ins = self.nodes[ch].insertion
            init_insertion = set(ins) == {0.0}  # are all ins values equal to 0?

            self.nodes[ch].heuristic.calculate_cost_from_parent(parent_heuristic, reset=True,
                                                                init_insertion=init_insertion)
            self.reset_heuristic_all_children(ch)

    def swap_parents(self, current_ind, new_parent_ind, new_heuristic):
        """Give a new parent to this node.

        Parameters
        ----------
        current_ind : int
            Index of the node of interest
        new_parent_ind : int
            Index of the new parent node
        new_heuristic : Heuristic
            Heuristic to use with the new parent

        Returns
        -------
        VOID
        """
        previous_parent = self.nodes[current_ind].parent
        new_parent_heuristic = self.nodes[new_parent_ind].heuristic
        self.nodes[current_ind].heuristic = new_heuristic
        self.nodes[previous_parent].children.remove(current_ind)
        self.nodes[current_ind].parent = new_parent_ind
        self.nodes[new_parent_ind].children.append(current_ind)

        ins = self.nodes[current_ind].insertion
        init_insertion = set(ins) == {0.0}

        new_heuristic.calculate_cost_from_parent(new_parent_heuristic, reset=True,
                                                 init_insertion=init_insertion)
        self.reset_heuristic_all_children(current_ind)

    def get_costs(self, child_ind):
        """Retrieves list of costs from child to root parent

        Parameters
        ----------
        child_ind : int
            index of node to start traversal

        Returns
        -------
        [float]
            costs of each node from the given index to the root
        """

        i = child_ind
        costs = []
        while True:
            costs.append(self.nodes[i].get_cost())
            if i == 0:
                break
            i = self.nodes[i].parent
        return costs

    def get_tube_data(self, child_ind):
        """Retrieves list of data (insertion/rotation) from **child to root**

        Parameters
        ----------
        child_ind : int
            index of node to start traversal

        Returns
        -------
        (list[list[float]]; list[list[float]]; list[list[int]])
            list of the insertion data from child to root
            list of the rotation data from child to root
            corresponds to index for "origin" SE3 array for each g_curve tube
        """

        index_list = self.get_index_list(child_ind)
        insertion = [self.nodes[i].insertion for i in index_list]
        rotation = [self.nodes[i].rotation for i in index_list]
        insert_indices = [self.nodes[i].insert_indices for i in index_list]

        return insertion, rotation, insert_indices

    def get_index_list(self, child_ind):
        """Gets a list of all the node indices from this child to root

        Parameters
        ----------
        child_ind : int
            Index of child

        Returns
        -------
        list[int]
            List of all the nodes from this child to root
        """
        i = child_ind
        index_list = [i]
        while i != 0:
            i = self.nodes[i].parent
            index_list.append(i)

        return index_list

    def get_tube_curves(self, child_ind):
        """Retrieves list of g_curves from **child to root**

        Parameters
        ----------
        child_ind : int
            index of node to start traversal

        Returns
        -------
        list of list of list of 4x4 arrays
            list of the curve data from child to root
        """

        i = child_ind
        g_out = [self.nodes[i].g_curves]
        while i != 0:
            i = self.nodes[i].parent
            g_out.append(self.nodes[i].g_curves)

        return g_out

    def save_tree(self, output_dir):
        """Save the tree in a text file, saved as tree.txt.

        Parameters
        ----------
        output_dir : Posix.Path
            Directory path for the tree file (not including pathname)

        Returns
        -------
        VOID
        """
        filename = output_dir / "tree.txt"
        tree_file = open(filename, "w")

        for i in range(len(self.nodes)):
            ins_str = ", ".join(str(ins) for ins in self.nodes[i].insertion)
            rot_str = ", ".join(str(rot) for rot in self.nodes[i].rotation)
            parent_str = self.nodes[i].parent
            if self.at_goal.__contains__(i):
                at_goal = True
            else:
                at_goal = False
            cost = self.nodes[i].heuristic.get_cost()
            this_node_str = f"{i} | {ins_str} | {rot_str} | {parent_str} | {cost} | {at_goal}\n"
            tree_file.write(this_node_str)

        tree_file.close()


class Node:
    """Each node holds the configuration data, as well as cost and tree refs."""

    def __init__(self, insertion, rotation, dim, heuristic, g_curves, insert_indices, parent=None):
        """
        Parameters
        ----------
        insertion : list[float]
            insertion values
        rotation: list[float]
            rotation values
        dim : int
            number of dimensions tree has for error checking
        heuristic : Heuristic
            Heuristic object containing cost information
        g_curves : list of list of 4x4 arrays
            g_curve[tube num][index] = 4x4 SE3 array
        insert_indices : list of int
            corresponds to index for "origin" SE3 array for each g_curve tube
        parent : int
            index of parent in node list (default is none)

        Raises
        ------
        Value Error
            If length of insertion or rotation does not match the given dim
        """
        if len(insertion) != dim or len(rotation) != dim:
            raise ValueError(f"Data size does not match given tube number ({dim})")
        self.insertion = insertion
        self.rotation = rotation
        self.parent = parent
        self.heuristic = heuristic
        self.g_curves = g_curves
        self.insert_indices = insert_indices
        self.children = []
        """list of int: positions of children in nodes list, set as children are inserted"""

    def get_cost(self):
        return self.heuristic.get_cost()
