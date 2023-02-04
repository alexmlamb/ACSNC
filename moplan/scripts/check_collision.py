import pathlib
import numpy as np

from ctrdapp.collision.collision_checker import CollisionChecker
from ctrdapp.solve.visualize_utils import visualize_curve_single

# todo deprecated
def main():

    path = pathlib.Path().absolute()
    objects_file = path / "configuration" / "big_sphere.json"
    collision_detector = CollisionChecker(objects_file)

    curve = [[np.eye(4), np.eye(4)]]
    curve[-1][-1][0, 3] = 15

    min_dist, goal = collision_detector.check_collision(curve, [1])

    print(min_dist)
    print("\ngoal: ")
    print(goal)

    visualize_curve_single(curve, objects_file, 1, [1], output_path, filename)


if __name__ == "__main__":
    main()
