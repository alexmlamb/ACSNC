import json
import pyvista as pv
import numpy as np
import time
from math import pi
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def parse_json(object_type, init_objects_file):
    """Parse through a environment json file to get obstacles and goal

    Parameters
    ----------
    object_type : str
        Class of object (either "obstacles" or "goal")
    init_objects_file : pathlib.Path
        json file declaring all the objects

    Returns
    -------

    """
    f = init_objects_file.open()
    full_json = json.load(f)
    f.close()

    list_objects = full_json.get(object_type)

    collision_objects = []
    for o in list_objects:
        if o.get('mesh'):
            mesh_file = init_objects_file.parent / o.get('filename')
            mesh_type = mesh_file.name.split('.')[-1]
            this_obj = pv.read(mesh_file, file_format=mesh_type.lower())  # todo figure out how to add mesh transform
        elif o.get('shape') == 'Sphere':
            this_obj = pv.Sphere(radius=o.get('radius'),
                                 center=o.get('transform'))
        elif o.get('shape') == 'Cylinder':
            this_obj = pv.Cylinder(radius=o.get('radius'),
                                   height=o.get('length'),
                                   center=o.get('transform'),
                                   direction=o.get('direction'))
        elif o.get('shape') == 'Box':
            x_half = o.get('x_length') / 2
            y_half = o.get('y_length') / 2
            z_half = o.get('z_length') / 2
            center = o.get('transform')
            this_obj = pv.Box([center[0] - x_half, center[0] + x_half,
                              center[1] - y_half, center[1] + y_half,
                              center[2] - z_half, center[2] + z_half])
        else:
            raise NotImplementedError(f'Shape type {o.get("shape")} not supported.')

        collision_objects.append(this_obj)

    return collision_objects


def visualize_curve(curve, objects_file, tube_num, tube_rad, output_dir, filename, visualize_from_indices=None,
                    user_adjust_view=True):
    """Save a movie showing progression of tubes in an environment

    Parameters
    ----------
    curve : list[list[list[np.ndarray]]]
        4X4 SE3 g values for each tube for each time point
    objects_file : pathlib.Path
        json file defining the objects
    tube_num : int
        Number of total tubes
    tube_rad : list[float]
        list of radii for the tubes
    output_dir : pathlib.Path
        directory to output the movie
    filename : str
        filename for the movie output
    visualize_from_indices: list[int]
        index from which to begin visualization for each tube; if whole tubes desired, None -> [0] * tube_num

    Returns
    -------
    VOID
    """
    if user_adjust_view:
        position_plot = pv.Plotter()
        add_objects(position_plot, objects_file)

        _ = add_single_curve(position_plot, curve[0], tube_num, tube_rad, visualize_from_indices)

        movie_cpos = position_plot.show()
    else:
        movie_cpos = [(0, 0, 0),
                      (0, 0, 0),
                      (0, 0, 0)]

    plotter = pv.Plotter()
    full_filename = output_dir / f"{filename}.mp4"
    plotter.open_movie(full_filename, framerate=3)  # todo make smoother

    add_objects(plotter, objects_file)

    plotter.show(auto_close=False, interactive_update=True, cpos=movie_cpos)
    plotter.write_frame()

    # plot each tube from root to final (only need p values, not R)
    # - plots of tubes must cut off the tube prior to insertion
    #   in movie form or all in one picture?
    for i in range(len(curve)):
        this_time_step = curve[i]
        this_tube_actor = add_single_curve(plotter, this_time_step, tube_num, tube_rad, visualize_from_indices)
        plotter.write_frame()
        plotter.update()
        time.sleep(0.3)
        for a in this_tube_actor:
            plotter.remove_actor(a)

    plotter.close()


def visualize_curve_single(curve, objects_file, tube_num, tube_rad, output_dir, filename, visualize_from_indices=None,
                           user_adjust_view=True, camera_angle=None, old_plotter=None, tube_color='g'):
    """Save a picture of tubes in an environment at a certain time point

    curve : list[list[np.ndarray]]
        4X4 SE3 g values for each tube
    objects_file : pathlib.Path
        json file defining the objects
    tube_num : int
        Number of total tubes
    tube_rad : list[float]
        list of radii for the tubes
    output_dir : pathlib.Path
        directory to output the movie
    filename : str
        filename for the movie output
    visualize_from_indices: list[int]
        index from which to begin visualization for each tube; if whole tubes desired, None -> [0] * tube_num

    Returns
    -------

    """
    if old_plotter is None:
        plotter = pv.Plotter()
        add_objects(plotter, objects_file)
    else:
        plotter = old_plotter

    _ = add_single_curve(plotter, curve, tube_num, tube_rad, visualize_from_indices, color=tube_color)

    full_filename = output_dir / f"{filename}.pdf"
    plotter.save_graphic(full_filename)

    # todo how to change camera angle interactively without closing the plotter??
    # if user_adjust_view:
    #     camera_angle = plotter.show()  # when this line returns, plotter is closed
    #     visualize_curve_single(curve, objects_file, tube_num, tube_rad, output_dir, filename,
    #                            visualize_from_indices, user_adjust_view=False, camera_angle=camera_angle)
    # else:
    #     full_filename = output_dir / f"{filename}.pdf"
    #     plotter.camera_position = camera_angle
    #     plotter.save_graphic(full_filename)

    return plotter




# todo remove visualize_from_indices
def add_single_curve(plotter, curve, tube_num, tube_rad, visualize_from_indices, color='w', invert_insert=False):
    """Helper to add a single set of curves to a pyvista Plotter

    Parameters
    ----------
    plotter : pv.Plotter
        The plotter object
    curve : list[list[np.ndarray]]
        4X4 SE3 g values for each tube
    tube_num : int
        Number of total tubes
    tube_rad : list[float]
        list of radii for the tubes
    visualize_from_indices: list[int]
        index from which to begin visualization for each tube; if whole tubes desired, None -> [0] * tube_num
    color : str
        Color compatible with pyvista color scheme
    invert_insert : bool
        Should the insertion values be inverted?

    Returns
    -------
    list[pv.Plotter.actor]
        list of actors resulting from the Plotter.add_mesh method
    """
    if not visualize_from_indices:  # todo remove visualize_from_indices
        visualize_from_indices = [0] * tube_num  # plot each whole tube

    this_tube_actor = []
    for n in range(tube_num):
        this_tube = curve[n]
        this_radius = float(tube_rad[n])

        # get the insertion index
        if invert_insert:
            index = len(this_tube) - visualize_from_indices[n]  # check todo
        else:
            index = visualize_from_indices[n]
        # get the P values from the insertion index to the "tip", with given radius
        these_p = [g[0:3, 3] for g in this_tube[index:]]
        if len(these_p) == 1:
            continue
            # todo this_tube_visual = pv.Cylinder(center=these_p[1], direction=)

        else:
            this_tube_spline = pv.Spline(np.array(these_p), len(these_p) * 20)
            this_tube_spline["scalars"] = np.arange(this_tube_spline.n_points)
            this_tube_visual = this_tube_spline.tube(radius=this_radius)
            this_tube_actor.append(plotter.add_mesh(this_tube_visual, color=color))

    return this_tube_actor
    # as movie or picture todo


def add_objects(plotter, objects_file):
    """Add the obstacles and goal to the given plotter

    Parameters
    ----------
    plotter : pv.Plotter
        The pyvista plotter for this scene
    objects_file : pathlib.Path
        Json file containing the object definitions

    Returns
    -------
    VOID
    """
    # plot obstacle mesh (with transparency)
    obstacle_meshes = parse_json('obstacles', objects_file)
    for o_m in obstacle_meshes:
        plotter.add_mesh(o_m, opacity=0.5)

    # plot goal mesh (should only be one)
    goal_meshes = parse_json('goal', objects_file)
    for g_m in goal_meshes:
        plotter.add_mesh(g_m, color='b', opacity=0.3)

    # plot insertion plane
    plane = pv.Plane(direction=[1, 0, 0], i_size=30, j_size=30)
    plotter.add_mesh(plane, color='tan', opacity=0.4)


def visualize_tree(from_points, to_points, node_list, output_dir, filename, solution_list, at_goal_list, cost_list):
    """Visualize a tree from the given point data, solution, and cost information for a single tube

    Parameters
    ----------
    from_points : list[list[float]]
        List of from data ([insertion, rotation])
    to_points : list[list[float]]
        List of to data ([insertion, rotation])
    node_list : list[int]
        List of node indices
    output_dir : pathlib.Path
        Directory for output
    filename : str
        Name for output file
    solution_list : list[int]
        List of indices from root to final node of the lowest cost solution
    at_goal_list : list[int]
        List of indices that reached the goal
    cost_list : list[float]
        List of node costs

    Returns
    -------
    VOID
    """
    fig, ax = plt.subplots()

    path_data = []
    solution_data = []
    goal_data = []
    to_data = []
    for from_pt, to_pt, node in zip(from_points, to_points, node_list):  # node index points to the TO node
        solution_node = solution_list.__contains__(node)
        goal_node = at_goal_list.__contains__(node)

        if from_pt[1] > pi:
            from_pt = [from_pt[0], from_pt[1] - 2*pi]
        if to_pt[1] > pi:
            to_pt = [to_pt[0], to_pt[1] - 2 * pi]

        to_data.append(to_pt)

        if abs(to_pt[1] - from_pt[1]) > pi:
            if from_pt[1] > to_pt[1]:
                pair_one = ([from_pt[0], from_pt[1] - 2*pi], to_pt)
                pair_two = (from_pt, [to_pt[0], to_pt[1] + 2*pi])
            else:
                pair_one = ([from_pt[0], from_pt[1] + 2 * pi], to_pt)
                pair_two = (from_pt, [to_pt[0], to_pt[1] - 2 * pi])

            path_data.append((mpath.Path.MOVETO, pair_one[0]))
            path_data.append((mpath.Path.LINETO, pair_one[1]))
            path_data.append((mpath.Path.MOVETO, pair_two[0]))
            path_data.append((mpath.Path.LINETO, pair_two[1]))

            if solution_node:
                solution_data.append((mpath.Path.MOVETO, pair_one[0]))
                solution_data.append((mpath.Path.LINETO, pair_one[1]))
                solution_data.append((mpath.Path.MOVETO, pair_two[0]))
                solution_data.append((mpath.Path.LINETO, pair_two[1]))
            if goal_node:
                goal_data.append((mpath.Path.MOVETO, pair_one[1]))
                goal_data.append((mpath.Path.MOVETO, pair_two[1]))
        else:
            path_data.append((mpath.Path.MOVETO, from_pt))
            path_data.append((mpath.Path.LINETO, to_pt))

            if solution_node:
                solution_data.append((mpath.Path.MOVETO, from_pt))
                solution_data.append((mpath.Path.LINETO, to_pt))
            if goal_node:
                goal_data.append((mpath.Path.MOVETO, to_pt))

    # plot lines between points
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, lw=0.5, color='0.6', zorder=1)
    ax.add_patch(patch)

    # plot solution lines
    codes, verts = zip(*solution_data)
    solution_path = mpath.Path(verts, codes)
    solution_patch = mpatches.PathPatch(solution_path, lw=0.6, color='r', zorder=2)
    ax.add_patch(solution_patch)

    # plot points
    c = [cost / max(cost_list) for cost in cost_list]
    x = [coord[0] for coord in to_data]
    y = [coord[1] for coord in to_data]
    ax.scatter(x, y, c=c, cmap='viridis_r', zorder=4, s=0.3)

    # plot at_goal points
    if goal_data:
        codes, verts = zip(*goal_data)
        goal_path = mpath.Path(verts, codes)
        x_goal, y_goal = zip(*goal_path.vertices)
        ax.scatter(x_goal, y_goal, color='k', s=2, zorder=3)

    ax.grid()
    ax.set_ylim(-pi, pi)

    output_full = output_dir / f"{filename}.pdf"
    plt.savefig(output_full)
