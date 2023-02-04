from ctrdapp.model.kinematic import save_g_positions
from ctrdapp.model.model_factory import create_model
from ctrdapp.solve.visualize_utils import visualize_curve_single
from ctrdapp.scripts.setup_script import setup_script

from math import pi
import numpy as np


def main():
    plotter = None
    i = 0
    save_curves = input("Save ng_curves in txt file? (y/n)")

    while True:
        collision_detector, objects_file, heuristic_factory, this_q, configuration, output_path = setup_script()

        # create model
        this_model = create_model(configuration, this_q)

        # todo get input from user
        this_g = this_model.solve_g(indices=[3, 9], thetas=[3.6609270062908728, 1.3051197304814763], full=False)

        if save_curves == 'y' or save_curves == 'yes':
            g_filename = output_path / f"g_curves_{i}.txt"
            save_g_positions(this_g, g_filename)

        t_col = input("Tube color for this run: ")
        plotter = visualize_curve_single(this_g, objects_file, configuration.get("tube_number"),
                                         configuration.get("tube_radius").get("outer"), output_path, "curve_visual",
                                         old_plotter=plotter, tube_color=t_col)

        addl_tube = input('Is viewing an additional tube desired? (y/n)')
        if addl_tube != 'y' or addl_tube != 'yes':
            break


if __name__ == "__main__":
    main()
