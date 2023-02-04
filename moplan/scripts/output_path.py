from ctrdapp.model.model_factory import create_model
from ctrdapp.solve.visualize_utils import visualize_curve
from ctrdapp.scripts.setup_script import setup_script


def output_path():
    """Script allows for visualizing tubes in time with specified insertion/rotations.

    Returns
    -------

    """
    collision_detector, objects_file, heuristic_factory, this_q, configuration, output_path = setup_script()

    this_model = create_model(configuration, this_q)

    # this is the g_curve for the initial position
    all_curves = [this_model.solve_g(indices=[7, 15], thetas=[0, 0], full=False)]

    # this iterates through many configurations; can be changed to get desired motion
    for i in range(1, 250):
        g_out, eta_out, new_insert_indices, true_insertions, ftl_out = \
            this_model.solve_integrate([0, 0.01], [0, 0], [0, 0.01 * i], [35, 75], all_curves[i - 1], invert_insert=False)

        all_curves.append(g_out)

    rad = configuration.get("tube_radius")
    visualize_curve(all_curves, objects_file, configuration.get("tube_number"), rad.get("outer"),
                    output_path, "FTL_linear_insertion_animated")


if __name__ == "__main__":
    output_path()
