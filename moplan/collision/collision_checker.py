"""Contains the CollisionChecker class for collision queries."""

import numpy as np
import fcl  # package name is in Pipfile "python-fcl" or "python-fcl-win32"
from .init_collision import add_goal, add_obstacles
from numpy.linalg import norm


class CollisionChecker:
    """Holds obstacles and goal and allows for collision queries.

    The CollisionChecker is initialized with obstacles (defined as basic
    shapes or as a mesh) and a goal. The initialized shapes are based on the
    init_collision.py file. Then, a curve (list of x, y, z coordinates) can be
    given to check for collision with obstacles and with the goal. This class
    serves as a convenient interface to work with the python-fcl wrapper.

    Parameters
    ----------
    init_objects_file : pathlib.PosixPath
        path to the json file describing the obstacles and goal

    Attributes
    ----------
    obstacles : list of fcl.CollisionObjects
        obstacle objects in the environment
    goal : fcl.CollisionObject
        goal object in the environment
    """

    def __init__(self, init_objects_file):
        self.obstacles = add_obstacles(init_objects_file)
        self.goal = add_goal(init_objects_file)

    def check_collision(self, curve, rad):
        """Determine if the curve given collides with obstacles or goal.

        The curve is translated to discretized cylinders with the given radius
        and checked for collisions with the obstacles and goal. Minimum distance
        to obstacles and tip distance to goal are returned, with a negative
        value denoting a collision.

        Parameters
        ----------
        curve : list[list[numpy.array]]
            list of 4x4 SE3 for each curve, with points given in last column
        rad : list[float]
            radii of the tubes

        Returns
        -------
        (float, float)
            minimum distance between curve and (obstacles, goal).
        """

        tube = self._build_tube(curve, rad)
        tube_manager = fcl.DynamicAABBTreeCollisionManager()
        tube_manager.registerObjects(tube)
        tube_manager.setup()
        obstacle_min = self._distance_check(tube_manager, self.obstacles)

        s = fcl.Sphere(rad[-1])  # creates a sphere with radius of last tube
        final_point = curve[-1][-1][0:3, 3]
        t = fcl.Transform(final_point)  # coordinates of last point of tube
        tip = fcl.CollisionObject(s, t)
        request = fcl.DistanceRequest()
        result = fcl.DistanceResult()
        goal_dist = fcl.distance(tip, self.goal, request, result)

        return obstacle_min, goal_dist

    @staticmethod
    def _distance_check(tube_manager, environment):
        """Checks distance between given collision manager and object list.

        Parameters
        ----------
        tube_manager : fcl.DynamicAABBTreeCollisionManager
            the collision manager with all tubes
        environment : list[fcl.CollisionObjects]
            list of collision objects

        Returns
        -------
        float
            minimum distance between the two collections of collision objects
        """

        env_manager = fcl.DynamicAABBTreeCollisionManager()
        env_manager.registerObjects(environment)
        env_manager.setup()
        data = fcl.DistanceData()
        tube_manager.distance(env_manager, data, fcl.defaultDistanceCallback)

        return data.result.min_distance

    @staticmethod
    def _build_tube(curve, rad):
        """Generates object list from given curve and radius lists.

        Creates a cylinder with given radius between every pair of [x, y, z]
        points. Each cylinder is initialized with the given radius and the
        computed length and given a transformation to move it from the origin
        (cylinders are initially centered on the origin in every axis) to the
        points, where the points are in the center of each face of the cylinder.

        Parameters
        ----------
        curve : list[list[numpy.array]]
            list of 4x4 SE3 for each curve, with points given in last column
        rad : list[float]
            radii of the tubes

        Returns
        -------
        list[fcl.CollisionObject]
            collection of cylinders that discretize the curve/tube
        """

        tube = []
        for n in range(len(curve)):
            for index, from_g in enumerate(curve[n][:-1]):
                to_g = curve[n][index + 1]

                from_point = from_g[0:3, 3]
                to_point = to_g[0:3, 3]

                vec = [(t - f) for f, t in zip(from_point, to_point)]
                mid_point = [(f + v/2) for f, v in zip(from_point, vec)]
                length = norm(vec)
                unit_vec = vec / length

                cyl = fcl.Cylinder(rad[n], length)

                if unit_vec[2] == -1.0:  # if vector is in -z direction
                    unit_quaternion = [0, 1, 0, 0]  # gives 180 degree rotation
                else:
                    quaternion = [1 + unit_vec[2], -unit_vec[1], unit_vec[0], 0]
                    quaternion_magnitude = norm(quaternion)
                    unit_quaternion = [q / quaternion_magnitude for q in quaternion]

                translate = np.array(mid_point)
                rotate = np.array(unit_quaternion)
                transform = fcl.Transform(rotate, translate)

                obj = fcl.CollisionObject(cyl, transform)
                tube.append(obj)
        return tube
