from .rrt import RRT
from .step import get_delta_rotation
from numpy.linalg import norm
from math import pi


class RRTStar(RRT):

    def __init__(self, model, heuristic_factory, collision_detector, configuration):
        super().__init__(model, heuristic_factory, collision_detector, configuration)
        self.single_tube_control = False  # impossible to maintain with the rewire operation

    def _single_iteration(self):
        delta_rotation, delta_insert, new_rotation, new_insert, original_neighbor_index, g_original_neighbor = self.calc_new_random_config()
        this_g, this_eta, insert_indices, true_insertion, ftl_neighbor = self.model.solve_integrate(delta_rotation,
                                                                                                    delta_insert,
                                                                                                    new_rotation,
                                                                                                    new_insert,
                                                                                                    g_original_neighbor)
        obs_min, goal_dist = self.cd.check_collision(this_g, self.tube_rad)
        if obs_min < 0:  # collision detected
            pass
        else:
            new_index = len(self.tree.nodes)  # before insertion, so no -1
            if goal_dist < 0:
                self.tree.at_goal.append(new_index)
                self.found_solution = True
                goal_dist = 0
            insert_fractions = [1 - (float(i) / ins_max) for i, ins_max in zip(true_insertion, self.insert_max)]  # inverse insertion
            insert_frac = sum(insert_fractions)  # = 0 if all tubes are fully inserted; = tube_num if fully retracted

            # point inserted into tree must be in format [insertion, rotation, insertion, rotation, ...]
            query_pt = true_insertion + new_rotation
            query_pt[::2] = true_insertion
            query_pt[1::2] = new_rotation

            # all neighbors of the new point within the bound
            neighbor_indices = self.tree.find_all_nearest_neighbor(query_pt, self.step_bound)  # think about limit todo

            cost_arr = []
            heur_arr = []

            for neighbor_ind in neighbor_indices:
                # any parent-dependent heuristics should be calculated here, only if necessary (fewer integration calls)
                if True and neighbor_ind != original_neighbor_index:  # todo only for parent-dependent heuristic
                    ins_neighbor = self.tree.nodes[neighbor_ind].insertion
                    rot_neighbor = self.tree.nodes[neighbor_ind].rotation
                    g_neighbor = self.tree.nodes[neighbor_ind].g_curves
                    delta_ins_neighbor = [t - p for t, p in zip(true_insertion, ins_neighbor)]
                    delta_rot_neighbor = [get_delta_rotation(p, t) for p, t in zip(rot_neighbor, new_rotation)]
                    _, _, _, _, ftl_neighbor = self.model.solve_integrate(delta_rot_neighbor, delta_ins_neighbor,
                                                                          new_rotation, true_insertion, g_neighbor,
                                                                          need_g_out=False)

                new_neighbor_heuristic = self.heuristic_factory.create(min_obstacle_distance=obs_min,
                                                                       goal_distance=goal_dist,
                                                                       follow_the_leader=ftl_neighbor,
                                                                       insertion_fraction=insert_frac)
                this_cost = new_neighbor_heuristic.test_cost_from_parent(self.tree.nodes[neighbor_ind].heuristic)
                cost_arr.append(this_cost)
                heur_arr.append(new_neighbor_heuristic)

            index_min = min(range(len(cost_arr)), key=cost_arr.__getitem__)
            best_neighbor_as_parent = neighbor_indices[index_min]
            best_heuristic = heur_arr[index_min]

            self.tree.insert(true_insertion, new_rotation, best_neighbor_as_parent,
                             best_heuristic, this_g, insert_indices)  # todo implement lazy insert/collision check

            neighbor_indices.remove(best_neighbor_as_parent)
            for ind in neighbor_indices:
                current_heuristic = self.tree.nodes[ind].heuristic
                this_cost = current_heuristic.get_cost()
                try:
                    no_cycle = self.tree.no_cycle(new_index, ind)
                except RecursionError:
                    print(f'Recursion not working on parent: {new_index} and child: {ind}')
                else:
                    if no_cycle:
                        child_ins = self.tree.nodes[ind].insertion
                        child_rot = self.tree.nodes[ind].rotation
                        g_parent = this_g
                        delta_ins_neighbor = [to - prev for to, prev in zip(child_ins, true_insertion)]
                        delta_rot_neighbor = [get_delta_rotation(prev, to) for to, prev in zip(child_rot, new_rotation)]

                        _, _, _, _, ftl_neighbor = self.model.solve_integrate(delta_rot_neighbor, delta_ins_neighbor,
                                                                              child_rot, child_ins, g_parent, need_g_out=False)

                        new_neighbor_heuristic = self.heuristic_factory.create_from_old(current_heuristic,
                                                                                        follow_the_leader=ftl_neighbor)
                        this_neighbor_cost = new_neighbor_heuristic.test_cost_from_parent(best_heuristic)
                        if this_neighbor_cost < this_cost:
                            self.tree.swap_parents(ind, new_index, new_neighbor_heuristic)
