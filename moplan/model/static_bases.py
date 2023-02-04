"""Basis for static equilibrium, defining tube shapes."""

import numpy as np

from ctrdapp.model.static import eye_from_theta
from ctrdapp.model.matrix_utils import big_adjoint


class StaticBasis:

    def __init__(self, delta_x, degree, strain_base, tube_num, q_dof, basis_type, output_q):
        self.delta_x = delta_x
        self.degree = degree
        self.strain_base = strain_base
        self.tube_num = tube_num
        self.q_dof = q_dof
        self.basis_type = basis_type
        self.q = output_q

    def get_basis(self, section_index, this_tube_num, tube_section, section_indices, theta):
        """Get the full basis for the given parameters.

        Parameters
        ----------
        section_index : int
            the index of this section, used for all overlapping tube bases
        this_tube_num : int
            the number of this tube (3 corresponds to the smallest tube and 2 to
            the second smallest, regardless of total tube number)
        tube_section : int
            the required section (again, 3 -> smallest tube)
        theta : list[float]
            theta values for the tubes, used for the "all_strain_bases" basis

        Returns
        -------
        np.ndarray
            the desired basis

        Notes
        -----
        Can be extended to allow for different types of bases.
        """
        if self.basis_type == 'simple':
            base = self._simple_basis(section_index, this_tube_num, tube_section)
        elif self.basis_type == 'last_strain_base':
            base = self._easy_last_basis(section_index, this_tube_num, tube_section, section_indices)
        elif self.basis_type == 'all_strain_bases':
            gg31 = eye_from_theta(theta[-1])
            adj = {3: big_adjoint(gg31)}
            if self.tube_num == 3:
                gg21 = eye_from_theta(theta[-2])
                adj[2] = {big_adjoint(gg21)}

            base = self._base_in_each_basis(section_index, this_tube_num, tube_section, adj, section_indices)
        else:
            raise NameError(f'{self.basis_type} is not a defined static basis name. Change config file'
                            f'to one of: simple, easy_last, or base_in_each')
        return base

    def _simple_basis(self, section_index, this_tube_num, tube_section):
        section_x_val = section_index * self.delta_x

        degree_adjusted = self.degree + 1
        if this_tube_num == tube_section:
            base = np.zeros((6, degree_adjusted * 3))
            for i in range(degree_adjusted):
                base[0, i] = base[1, i + 3] = base[2, i + 6] = section_x_val ** i
        else:
            base = np.zeros((6, degree_adjusted))
            for i in range(degree_adjusted):
                base[0, i] = section_x_val ** i

        return base

    def _easy_last_basis(self, section_index, this_tube_num, tube_section, section_indices):
        if this_tube_num == tube_section == 3:
            true_x_val = (section_index + section_indices.get((this_tube_num, tube_section))) * self.delta_x
            this_strain_base = self.strain_base[this_tube_num + self.tube_num - 4]
            this_q_dof = self.q_dof[self.tube_num - 1]
            base = this_strain_base(true_x_val, this_q_dof)
        else:
            base = self._simple_basis(section_index, this_tube_num, tube_section)

        return base

    def _base_in_each_basis(self, section_index, this_tube_num, tube_section, adj, section_indices):
        section_x_val = section_index * self.delta_x
        if this_tube_num == tube_section:
            x_val_3 = (section_index + section_indices.get((3, tube_section))) * self.delta_x
            base3 = self.strain_base[-1](x_val_3, self.q_dof[-1])
            if this_tube_num == 3:
                base = base3
            else:
                x_val_2 = (section_index + section_indices.get((2, tube_section))) * self.delta_x
                base2 = self.strain_base[-2](x_val_2, self.q_dof[-2])

                if this_tube_num == 2:
                    base = np.zeros((6, self.degree + 3))
                    for i in range(self.degree + 1):
                        base[0, i] = section_x_val ** i
                    base[:, self.degree + 2] = base2
                    base[:, self.degree + 3] = adj.get(3) @ base3
                else:  # (this_tube_num == 1)
                    x_val_1 = (section_index + section_indices.get((1, tube_section))) * self.delta_x
                    base1 = self.strain_base[-3](x_val_1, self.q_dof[-3])

                    base = np.zeros((6, self.degree + 4))
                    for i in range(self.degree + 1):
                        base[0, i] = section_x_val ** i
                    base[:, self.degree + 2] = base1
                    base[:, self.degree + 3] = adj.get(2) @ base2
                    base[:, self.degree + 4] = adj.get(2) @ adj.get(3) @ base3

        else:
            base = self._simple_basis(section_index, this_tube_num, tube_section)

        return base

    def get_static_dof(self):
        section_indices = {(1, 1): 1, (2, 1): 1, (3, 1): 1,
                           (2, 2): 1, (3, 2): 1, (3, 3): 1}

        if self.tube_num == 3:
            theta = [0, 0, 0]

            dof_11 = len(self.get_basis(1, 1, 1, section_indices, theta)[0])
            dof_21 = len(self.get_basis(1, 2, 1, section_indices, theta)[0])
            dof_31 = len(self.get_basis(1, 3, 1, section_indices, theta)[0])
        elif self.tube_num == 2:
            dof_11 = dof_21 = dof_31 = 0
            theta = [0, 0]
        else:
            raise ValueError(f'The static model does not support {self.tube_num} tubes.')
        dof_22 = len(self.get_basis(1, 2, 2, section_indices, theta)[0])
        dof_32 = len(self.get_basis(1, 3, 2, section_indices, theta)[0])
        dof_33 = len(self.get_basis(1, 3, 3, section_indices, theta)[0])

        return {(1, 1): dof_11, (2, 1): dof_21, (3, 1): dof_31,
                (2, 2): dof_22, (3, 2): dof_32, (3, 3): dof_33}

    def get_init_vals(self):
        ndof = self.get_static_dof()
        init_guess = np.zeros(sum(ndof.values()) - ndof.get((3, 3)))

        set_value = 0.01
        degree_size = self.degree + 1

        if self.basis_type == 'all_strain_bases':  # todo fix for two-tube
            # (Degree, 1, 1, 1, Degree, Degree), Degree, 1, 1, Degree, 3rd base

            indices = np.asarray([0, degree_size, degree_size + 1, degree_size + 2,
                                  degree_size + 3, degree_size * 2 + 3,
                                  degree_size * 3 + 3, degree_size * 3 + 4,
                                  degree_size * 3 + 5, degree_size * 3 + 6])
        else:
            first_tubes = sum(ndof.values()) - ndof.get((3, 3))
            indices = np.arange(0, first_tubes, degree_size)

        for ind in indices:
            init_guess[ind] = set_value
        if self.basis_type == 'all_strain_basis' or self.basis_type == 'last_strain_base':
            init_guess = np.append(init_guess, self.q[-1])
        else:
            init_guess = np.append(init_guess, [set_value])
            init_guess = np.append(init_guess, np.zeros(degree_size - 1))

        return init_guess


