from numpy.linalg import norm
from numpy import array, count_nonzero
from math import floor, pi


# todo CLEAN UP
def step(prev: [float], control_params: [float], max_bound: float) -> [float]:
    """Returns a point between prev and to, at most max_bound distance from prev

    If the distance between the points, given by the norm is less than the
    max_bound, the to list is returned.

    Parameters
    ----------
    prev : list of float
        the initial/from point, including only insertions
    control_params : list of float
        the goal/towards point, including both insertions and rotations
    max_bound : float
        the maximum distance to travel from prev

    Returns
    -------
    [float] : point with same dimension as prev and to

    Raises
    ------
    ValueError
        If the given points are of different dimensions
    """
    to = control_params[::2]
    if len(prev) != len(to):
        raise ValueError("Given vectors are not the same dimension")
    vec = []
    for p, t in zip(prev, to):
        vec.append(t - p)

    # if vector magnitude > max, multiply unit vector by max
    vec_mag = norm(vec)
    final = []
    if vec_mag > max_bound:
        for v, p in zip(vec, prev):
            final.append(v*max_bound/vec_mag + p)
        return final
    else:
        return to


def step_rotation(prev: [float], control_params: [float], rotation_max: float):
    """Calculate the next rotation coordinate and the angular velocity.

    Given the previous rotation coordinate and the control parameters (including both linear and
    angular velocities) and a max rotation scalar, calculates the final rotation.

    Parameters
    ----------
    prev : [float]
        Previous rotation values
    control_params : [float]
        the goal/towards point, including both insertions and rotations
    rotation_max : float
        Maximum rotation allowed

    Returns
    -------
    ([float]; [float])
        Coordinates of the final rotation; angular velocity between the previous rotation
        coordinates and the output (in each dimension)

    Raises
    ------
    ValueError
        If the given previous and to vectors aren't the same size
    """
    to = control_params[1::2]
    if len(prev) != len(to):
        raise ValueError("Given vectors are not the same dimension")
    vec = []
    vec_with_dir = []
    for p, t in zip(prev, to):
        velocity = get_delta_rotation(p, t)
        vec.append(abs(velocity))
        vec_with_dir.append(velocity)

    # if vector magnitude > max, multiply unit vector by max
    vec_mag = norm(vec)
    final = []
    final_delta = []
    if vec_mag > rotation_max:
        for v, p in zip(vec_with_dir, prev):
            delta = v * rotation_max / vec_mag
            val = delta + p
            final.append(val % (2 * pi))
            final_delta.append(delta)
        return final, final_delta
    else:
        return to, vec_with_dir


def get_delta_rotation(prev, to):
    """Get the delta between these two numbers, given they are on a circular manifold [0, 2*pi].

    Parameters
    ----------
    prev : float
        Previous coordinate
    to : float, [0, 2*pi]
        Destination coordinate, [0, 2*pi]

    Returns
    -------
    float
        the velocity between the two coordinates
    """
    magnitude = min(abs(to - prev), 2 * pi - abs(to - prev))
    if (to > prev and (to - prev) < pi) or (prev > to and (prev - to) > pi):
        direction = 1
    else:
        direction = -1

    return magnitude * direction


def get_single_tube_value(tube_control_params, insert_neighbor, rotation_neighbor, neighbor_parent, probability, random_num):
    """Returns an insertion vector with one changed value from insert_neighbor.

    Compares the insert_neighbor vector to its parent's vector to find the tube
    that is currently being inserted. The given probability is for selecting the
    same tube as previous and selecting another tube is
    (1 - probability) / tube_num. Output is the insert_neighbor vector with the
    insert_rand value for the randomly chosen tube (based on the probabilities).

    Parameters
    ----------
    tube_control_params : list of float
        The new insertion and rotation array [ins0, rot0, ins1, rot1, ...]
    insert_neighbor : list of float
        The insertions of the nearest neighbor of the tube_control_params array
    rotation_neighbor : list of float
        The rotations of the nearest neighbor
    neighbor_parent : list of float
        The parent of the insert_neighbor array
    probability : float
        The probability of selecting the same tube as previously changed
    random_num : float
        Random [0, 1) for tube selection (given as parameter for ease of testing)

    Returns
    -------
    list of float : the final insertion array
        combination of insert_neighbor and insert_rand
    int : the tube number with a delta insertion
    """
    tube_num = len(insert_neighbor)
    if tube_num == 1:
        return tube_control_params, 0
    else:
        neighbor_diff = array([n - p for n, p in zip(insert_neighbor, neighbor_parent)])
        if count_nonzero(neighbor_diff) > 1:
            raise ValueError("Multiple tubes inserted in neighbor node.")
        elif count_nonzero(neighbor_diff) == 0:
            last_insertion_tube_num = 0  # default to first tube (esp. for initial insertion)
        else:
            last_insertion_tube_num = neighbor_diff.nonzero()[0][0]
        possible_tubes = list(range(tube_num))
        possible_tubes.pop(last_insertion_tube_num)

        shifted_r = random_num - probability
        if shifted_r < 0:
            this_insertion_tube_num = last_insertion_tube_num
        else:
            probability_other = (1 - probability) / (tube_num - 1)
            this_insertion_tube_num = possible_tubes[floor(shifted_r/probability_other)]

        new_params = []
        for i in range(tube_num):
            if i == this_insertion_tube_num:
                new_params.append(tube_control_params[2*i])
                new_params.append(tube_control_params[2*i+1])
            else:
                new_params.append(insert_neighbor[i])
                new_params.append(rotation_neighbor[i])
        return new_params, this_insertion_tube_num






