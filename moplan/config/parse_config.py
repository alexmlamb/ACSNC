"""Parses a given configuration file."""

import yaml


def parse_config(file):
    """Parses given yaml file using requirements and defaults.

    Required parameters are defined in this function along with
    parameter defaults. Parameters are grouped by package name.

    Parameters
    ----------
    file : pathlib.PosixPath
        configuration file

    Returns
    -------
    (dict, dict)
        dictionary of configuration parameters;
        dictionaries of lists of key-names

    Notes
    -----
    This file must be kept up to date as new modules are added to
    different packages as well as new parameters being specified. To add
    a new parameter, add its name to the appropriate dictionary and to
    the defaults dictionary (with a suitable default. To add a
    new module (new type of solver, optimizer, model, or heuristic),
    add a new entry to the dictionaries below with a list of all
    the required parameters.
    """

    # dictionaries specifying required parameters for various modules
    general = ["tube_number", "tube_radius", "collision_objects_filename"]
    optimizers = {"nelder_mead": ["optimizer_precision", "optimize_iterations"]}
    solvers = {"rrt": ["step_bound", "iteration_number",
                       "tube_lengths", "single_tube_control", "rotation_max"],
               "rrt_star": ["step_bound", "iteration_number", "tube_lengths", "rotation_max"]}
    models = {"kinematic": ["q_dof", "delta_x", "tube_lengths", "strain_bases"],
              "static": ["q_dof", "delta_x", "tube_lengths", "strain_bases",
                         "basis_type", "degree"]}
    heuristics = {"square_obstacle_avg_plus_weighted_goal": ["goal_weight"],
                  "only_goal_distance": [],
                  "follow_the_leader": ["only_tip"],
                  "follow_the_leader_w_insertion": ["only_tip", "insertion_weight"],
                  "follow_the_leader_translation": ["only_tip"]}

    # groups together the required parameter dictionaries
    dictionaries = {"optimizer": optimizers, "solver": solvers,
                    "model": models, "heuristic": heuristics}

    # dictionary detailing all of the default parameter values
    defaults = {"optimizer_type": "nelder_mead",
                "solver_type": "rrt",
                "model_type": "kinematic",
                "heuristic_type": "square_obstacle_avg_plus_weighted_goal",
                "tube_number": 2,
                "tube_radius": [3, 2],
                "collision_objects_filename": "init_objects.json",
                "optimizer_precision": 0.1,
                "step_bound": 3,
                "tube_lengths": [60, 50],
                "iteration_number": 2000,
                "rewire_probability": 0.1,
                "goal_weight": 2,
                "q_dof": [1, 1],
                "delta_x": 1,
                "single_tube_control": True,
                "optimize_iterations": 50,
                "only_tip": True,
                "insertion_weight": 10,
                "strain_bases": ["constant", "constant"],
                "basis_type": "last_strain_base",
                "degree": 2,
                "rotation_max": 0.1745
                }

    with file.open(mode='r') as fid:
        config = yaml.full_load(fid)

    if config is None:  # for empty config file
        config = dict()

    for g in general:
        if g not in config:
            config[g] = defaults.get(g)
            print(f"{g} not specified in {file.name}. Using default value "
                  f"{defaults.get(g)} instead.")

    _config_helper("optimizer_type", optimizers, config, file.name, defaults)
    _config_helper("solver_type", solvers, config, file.name, defaults)
    _config_helper("model_type", models, config, file.name, defaults)
    _config_helper("heuristic_type", heuristics, config, file.name, defaults)

    config_validation(config)

    return config, dictionaries


def config_validation(configuration):
    """Validates and modifies the dictionary

    Checks tube number against given radii, max tube lengths,
    and given q_dof. Checks that the lengths are all divisible
    by delta_x and raises an error if not.

    Parameters
    ----------
    configuration : dict
        current dictionary of configuration parameters

    Returns
    -------
    dict
        adjusted dictionary for appropriate tube numbers

    Raises
    ------
    ValueError
        If given dictionary has incompatible types
    """
    tube_num = configuration.get('tube_number')
    q_dof = configuration.get('q_dof')
    radius = configuration.get('tube_radius')
    delta_x = configuration.get('delta_x')
    tube_lengths = configuration.get('tube_lengths')

    if isinstance(q_dof, int):
        configuration['q_dof'] = [q_dof] * tube_num
        print(f"Using {q_dof} as q_dof for every tube.\n")
    elif isinstance(q_dof, list) and len(q_dof) == tube_num:
        pass
    else:
        raise ValueError(f"Input for q_dof of {q_dof} is not suitable.\n")

    if isinstance(radius, list) and len(radius) == tube_num:
        inner = [rad - 0.1 for rad in radius]
        configuration['tube_radius'] = {'outer': radius, 'inner': inner}
    elif isinstance(radius, dict) and 'outer' in radius.keys() and len(radius.get('outer')) == tube_num:
        if 'inner' in radius.keys() and len(radius.get('inner')) == tube_num:
            pass
        else:
            radius['inner'] = [rad - 0.1 for rad in radius.get('outer')]
            configuration['tube_radius'] = radius
    else:
        raise ValueError(f"Input for radius of {radius} is not suitable.\n")

    if isinstance(tube_lengths, (int, float)):
        configuration['tube_lengths'] = [tube_lengths] * tube_num
        print(f"Using {tube_lengths} as length for every tube.\n")
    elif isinstance(tube_lengths, list) and len(tube_lengths) == tube_num:
        pass
    else:
        raise ValueError(f"Input for tube_lengths of {tube_lengths} is not suitable.\n")

    new_lengths = configuration.get('tube_lengths')
    for this_length in new_lengths:
        if this_length % delta_x != 0:
            raise ValueError(f"Length input {this_length} not divisible by delta_x: {delta_x}\n")


def _config_helper(config_key, required_dict, config, filename, defaults):
    """Helper function that sets parameters for given package type.

    Gets the list of required parameters for the module specified by the
    config_key entry in the config. Adds these parameters to the config,
    if they exist; if not, sets to the default values with a printed
    message detailing the omission.

    Parameters
    ----------
    config_key : str
        describes a package as one of four options: "optimizer_type",
        "solver_type", "model_type", "heuristic_type"
    required_dict : dict
        dictionary of required parameters for package described in config_key
    config : dict
        current dictionary of configuration parameters
    filename : str
        name of configuration file for error-message purposes
    defaults : dict
        dictionary with all the default parameter values

    Returns
    -------
    None
    """
    this_type = config.get(config_key)
    if this_type not in required_dict:
        config[config_key] = defaults.get(config_key)
        print(f"{config_key} {this_type} does not exist. Using default of "
              f"{defaults.get(config_key)} instead.")

    new_type = config.get(config_key)
    for c in required_dict.get(new_type):
        if c not in config:
            config[c] = defaults.get(c)
            print(f"{config_key} is {new_type} however {c} not specified "
                  f"in {filename}. Default is {defaults.get(c)}.")
