import pathlib
from time import strftime
import shutil

from ctrdapp.config.parse_config import parse_config
from ctrdapp.collision.collision_checker import CollisionChecker
from ctrdapp.heuristic.heuristic_factory import create_heuristic_factory


def setup_script():
    import sys  # todo global
    sys.setrecursionlimit(10 ** 4)

    path = pathlib.Path().absolute()
    config_filename = input("Enter name of configuration file (with .yaml extension): ")
    while True:
        file = path / "configuration" / config_filename
        if file.exists():
            break
        else:
            config_filename = input(f"{config_filename} doesn't exist. Please input configuration filename again: ")
    configuration, dictionaries = parse_config(file)

    run_name = configuration.get("run_identifier")
    output_path = path / "output" / run_name
    try:
        output_path.mkdir(parents=True)
    except FileExistsError:
        current_time = strftime("%H%M")
        current_date = strftime("%m%d%y")
        new_name = f"{run_name}_{current_date}_{current_time}"
        output_path = path / "output" / new_name
        output_path.mkdir(parents=True)

    shutil.copy(file, output_path / f"{run_name}.yaml")

    objects_file = path / "configuration" / configuration.get("collision_objects_filename")
    shutil.copy(objects_file, output_path / f"{run_name}.json")
    collision_detector = CollisionChecker(objects_file)

    # create heuristic factory
    heuristic_factory = create_heuristic_factory(configuration,
                                                 dictionaries.get("heuristic"))

    q_dof = configuration.get('q_dof')
    q_str = input(f"Enter desired q (or initial guess); should be {sum(q_dof)} "
                  f"numbers long, each separated by a comma. "
                  f"(Tubes have these degrees of freedom: {q_dof}).")

    while True:
        output_q = q_str.split(", ")
        output_q = [float(g) for g in output_q]
        if len(output_q) == sum(q_dof):
            break
        else:
            q_str = input(f"Incorrect format of q. Input {sum(q_dof)} numbers "
                          f"each separated by a comma: ")

    return collision_detector, objects_file, heuristic_factory, output_q, configuration, output_path


if __name__ == "__main__":
    setup_script()
