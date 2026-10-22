# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure necessary to handle mpi buffers for the particles
"""
import numpy as np
import numba
from typing import Any

# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed
from fbpic.utils.printing import catch_gpu_memory_error

if cuda_installed:
    import cupy
    import warp as wp
    from fbpic.utils.cuda import cuda_tpb_bpg_1d
    from fbpic.utils.warp import warp_kernel


def remove_outside_particles(species, fld, n_guard, left_proc, right_proc):
    """
    Remove the particles that are outside of the physical domain (i.e.
    in the guard cells). Store them in sending buffers, which are returned.

    Parameters
    ----------
    species: a Particles object
        Contains the data of this species

    fld: a Fields object
        Contains information about the dimension of the grid,
        and the prefix sum (when using the GPU)

    n_guard: int
        Number of guard cells

    left_proc, right_proc: int or None
        Indicate whether there is a left or right processor or if the
        boundary is open (None).

    Returns
    -------
    float_send_left, float_send_right, uint_send_left, uint_send_right:
        arrays of shape (n_float,Nptcl) and (n_int,Nptcl) where Nptcl
        is the number of particles that are sent to the left
        proc and right proc respectively, and where n_float and n_int
        are the number of float and integer quantities respectively
    If left_proc or right_proc is None, the corresponding array has Nptcl=0
    """
    if species.use_cuda:
        # Remove outside particles on GPU, and copy buffers on CPU
        float_send_left, float_send_right, uint_send_left, uint_send_right = (
            remove_particles_gpu(species, fld, n_guard, left_proc, right_proc)
        )
    else:
        # Remove outside particles on the CPU
        float_send_left, float_send_right, uint_send_left, uint_send_right = (
            remove_particles_cpu(species, fld, n_guard, left_proc, right_proc)
        )

    return (float_send_left, float_send_right, uint_send_left, uint_send_right)


def remove_particles_cpu(species, fld, n_guard, left_proc, right_proc):
    """
    Remove the particles that are outside of the physical domain (i.e.
    in the guard cells). Store them in sending buffers, which are returned.

    Parameters
    ----------
    species: a Particles object
        Contains the data of this species

    fld: a Fields object
        Contains information about the dimension of the grid,
        and the prefix sum (when using the GPU)

    n_guard: int
        Number of guard cells

    left_proc, right_proc: int or None
        Indicate whether there is a left or right processor or if the
        boundary is open (None).

    Returns
    -------
    float_send_left, float_send_right, uint_send_left, uint_send_right:
        arrays of shape (n_float,Nptcl) and (n_int,Nptcl) where Nptcl
        is the number of particles that are sent to the left
        proc and right proc respectively, and where n_float and n_int
        are the number of float and integer quantities respectively
    """
    # Calculate the positions between which to remove particles
    # For the open boundaries, only the particles in the outermost
    # half of the guard cells are removed
    zbox_min = fld.interp[0].zmin + n_guard * fld.interp[0].dz
    zbox_max = fld.interp[0].zmax - n_guard * fld.interp[0].dz

    # Select the particles that are in the left or right guard cells,
    # and those that stay on the local process
    selec_left = species.z < zbox_min
    selec_right = species.z > zbox_max
    selec_stay = np.logical_not(selec_left) & np.logical_not(selec_right)

    # Shortcuts
    n_float = species.n_float_quantities
    n_int = species.n_integer_quantities

    # Allocate and fill left sending buffer
    if left_proc is not None:
        N_send_l = selec_left.sum()
        float_send_left = np.empty((n_float, N_send_l), dtype=np.float64)
        uint_send_left = np.empty((n_int, N_send_l), dtype=np.uint64)
        float_send_left[0, :] = species.x[selec_left]
        float_send_left[1, :] = species.y[selec_left]
        float_send_left[2, :] = species.z[selec_left]
        float_send_left[3, :] = species.ux[selec_left]
        float_send_left[4, :] = species.uy[selec_left]
        float_send_left[5, :] = species.uz[selec_left]
        float_send_left[6, :] = species.inv_gamma[selec_left]
        float_send_left[7, :] = species.w[selec_left]
        i_attr = 0
        if species.tracker is not None:
            uint_send_left[i_attr, :] = species.tracker.id[selec_left]
            i_attr += 1
        if species.ionizer is not None:
            uint_send_left[i_attr, :] = species.ionizer.ionization_level[selec_left]
            float_send_left[8, :] = species.ionizer.w_times_level[selec_left]
        if species.spin_tracker is not None:
            float_send_left[-3, :] = species.spin_tracker.sx[selec_left]
            float_send_left[-2, :] = species.spin_tracker.sy[selec_left]
            float_send_left[-1, :] = species.spin_tracker.sz[selec_left]
    else:
        # No need to allocate and copy data ; return an empty array
        float_send_left = np.empty((n_float, 0), dtype=np.float64)
        uint_send_left = np.empty((n_int, 0), dtype=np.uint64)

    # Allocate and fill right sending buffer
    if right_proc is not None:
        N_send_r = selec_right.sum()
        float_send_right = np.empty((n_float, N_send_r), dtype=np.float64)
        uint_send_right = np.empty((n_int, N_send_r), dtype=np.float64)
        float_send_right[0, :] = species.x[selec_right]
        float_send_right[1, :] = species.y[selec_right]
        float_send_right[2, :] = species.z[selec_right]
        float_send_right[3, :] = species.ux[selec_right]
        float_send_right[4, :] = species.uy[selec_right]
        float_send_right[5, :] = species.uz[selec_right]
        float_send_right[6, :] = species.inv_gamma[selec_right]
        float_send_right[7, :] = species.w[selec_right]
        i_attr = 0
        if species.tracker is not None:
            uint_send_right[i_attr, :] = species.tracker.id[selec_right]
            i_attr += 1
        if species.ionizer is not None:
            uint_send_right[i_attr, :] = species.ionizer.ionization_level[selec_right]
            float_send_right[8, :] = species.ionizer.w_times_level[selec_right]
        if species.spin_tracker is not None:
            float_send_right[-3, :] = species.spin_tracker.sx[selec_right]
            float_send_right[-2, :] = species.spin_tracker.sy[selec_right]
            float_send_right[-1, :] = species.spin_tracker.sz[selec_right]
    else:
        # No need to allocate and copy data ; return an empty array
        float_send_right = np.empty((n_float, 0), dtype=np.float64)
        uint_send_right = np.empty((n_int, 0), dtype=np.float64)

    # Resize the particle arrays
    N_stay = selec_stay.sum()
    species.Ntot = N_stay
    species.x = species.x[selec_stay]
    species.y = species.y[selec_stay]
    species.z = species.z[selec_stay]
    species.ux = species.ux[selec_stay]
    species.uy = species.uy[selec_stay]
    species.uz = species.uz[selec_stay]
    species.inv_gamma = species.inv_gamma[selec_stay]
    species.w = species.w[selec_stay]
    if species.tracker is not None:
        species.tracker.id = species.tracker.id[selec_stay]
    if species.ionizer is not None:
        species.ionizer.w_times_level = species.ionizer.w_times_level[selec_stay]
        species.ionizer.ionization_level = species.ionizer.ionization_level[selec_stay]
    if species.spin_tracker is not None:
        species.spin_tracker.sx = species.spin_tracker.sx[selec_stay]
        species.spin_tracker.sy = species.spin_tracker.sy[selec_stay]
        species.spin_tracker.sz = species.spin_tracker.sz[selec_stay]

    # Return the sending buffers
    return (float_send_left, float_send_right, uint_send_left, uint_send_right)


@catch_gpu_memory_error
def remove_particles_gpu(species, fld, n_guard, left_proc, right_proc):
    """
    Remove the particles that are outside of the physical domain (i.e.
    in the guard cells). Store them in sending buffers, which are returned.

    Parameters
    ----------
    species: a Particles object
        Contains the data of this species

    fld: a Fields object
        Contains information about the dimension of the grid,
        and the prefix sum (when using the GPU)

    n_guard: int
        Number of guard cells

    left_proc, right_proc: int or None
        Indicate whether there is a left or right processor or if the
        boundary is open (None).

    Returns
    -------
    float_send_left, float_send_right, uint_send_left, uint_send_right:
        arrays of shape (n_float,Nptcl) and (n_int,Nptcl) where Nptcl
        is the number of particles that are sent to the left
        proc and right proc respectively, and where n_float and n_int
        are the number of float and integer quantities respectively
    """
    # Check if particles are sorted
    # (The particles are usually expected to be sorted from the previous
    # iteration at this point - except at the first iteration of `step`.)
    if not species.sorted:
        species.sort_particles(fld=fld)
        species.sorted = True

    # Get the particle indices between which to remove the particles
    # (Take into account the fact that the moving window may have
    # shifted the grid since the particles were last sorted: prefix_sum_shift)
    prefix_sum = species.prefix_sum
    Nz = fld.Nz
    Nr = fld.Nr
    # Find the z index of the first cell for which particles are kept
    iz_min = max(n_guard + species.prefix_sum_shift, 0)
    # Find the z index of the first cell for which particles are removed again
    iz_max = min(Nz - n_guard + species.prefix_sum_shift + 1, Nz)
    # Find the corresponding indices in the particle array
    # Reminder: prefix_sum[i] is the cumulative sum of the number of particles
    # in cells 0 to i (where cell i is included)
    if iz_min * (Nr + 1) - 1 >= 0:
        i_min = int(prefix_sum[iz_min * (Nr + 1) - 1])
    else:
        i_min = 0
    i_max = int(prefix_sum[iz_max * (Nr + 1) - 1])

    # Total number of particles in each particle group
    N_send_l = i_min
    new_Ntot = i_max - i_min
    N_send_r = species.Ntot - i_max

    # Allocate the sending buffers on the CPU
    n_float = species.n_float_quantities
    n_int = species.n_integer_quantities
    if left_proc is not None:
        float_send_left = np.empty((n_float, N_send_l), dtype=np.float64)
        uint_send_left = np.empty((n_int, N_send_l), dtype=np.uint64)
    else:
        float_send_left = np.empty((n_float, 0), dtype=np.float64)
        uint_send_left = np.empty((n_int, 0), dtype=np.uint64)
    if right_proc is not None:
        float_send_right = np.empty((n_float, N_send_r), dtype=np.float64)
        uint_send_right = np.empty((n_int, N_send_r), dtype=np.uint64)
    else:
        float_send_right = np.empty((n_float, 0), dtype=np.float64)
        uint_send_right = np.empty((n_int, 0), dtype=np.uint64)

    # Get the threads per block and the blocks per grid
    dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d(species.Ntot)
    # Float quantities:
    # Build list of float attributes to copy
    attr_list = [
        (species, "x"),
        (species, "y"),
        (species, "z"),
        (species, "ux"),
        (species, "uy"),
        (species, "uz"),
        (species, "inv_gamma"),
        (species, "w"),
    ]
    if species.ionizer is not None:
        attr_list.append((species.ionizer, "w_times_level"))
    if species.spin_tracker is not None:
        attr_list += [
            (species.spin_tracker, "sx"),
            (species.spin_tracker, "sy"),
            (species.spin_tracker, "sz"),
        ]
    # Loop through the float attributes
    for i_attr in range(n_float):
        # Initialize 3 buffer arrays on the GPU (need to be initialized
        # inside the loop, as `copy_to_host` invalidates these arrays)
        left_buffer = cupy.empty((N_send_l,), dtype=np.float64)
        right_buffer = cupy.empty((N_send_r,), dtype=np.float64)
        stay_buffer = cupy.empty((new_Ntot,), dtype=np.float64)
        # Check that the buffers are still on GPU
        # (safeguard against automatic memory management)
        assert type(left_buffer) is not np.ndarray
        assert type(right_buffer) is not np.ndarray
        assert type(left_buffer) is not np.ndarray
        # Split the particle array into the 3 buffers on the GPU
        particle_array = getattr(attr_list[i_attr][0], attr_list[i_attr][1])
        split_particles_to_buffers[dim_grid_1d, dim_block_1d](
            particle_array, left_buffer, stay_buffer, right_buffer, i_min, i_max
        )
        # Assign the stay_buffer to the initial particle data array
        # and fill the sending buffers (if needed for MPI)
        setattr(attr_list[i_attr][0], attr_list[i_attr][1], stay_buffer)
        if left_proc is not None:
            left_buffer.get(out=float_send_left[i_attr])
        if right_proc is not None:
            right_buffer.get(out=float_send_right[i_attr])

    # Integer quantities:
    if n_int > 0:
        attr_list = []
    if species.tracker is not None:
        attr_list.append((species.tracker, "id"))
    if species.ionizer is not None:
        attr_list.append((species.ionizer, "ionization_level"))
    for i_attr in range(n_int):
        # Initialize 3 buffer arrays on the GPU (need to be initialized
        # inside the loop, as `copy_to_host` invalidates these arrays)
        left_buffer = cupy.empty((N_send_l,), dtype=np.uint64)
        right_buffer = cupy.empty((N_send_r,), dtype=np.uint64)
        stay_buffer = cupy.empty((new_Ntot,), dtype=np.uint64)
        # Split the particle array into the 3 buffers on the GPU
        particle_array = getattr(attr_list[i_attr][0], attr_list[i_attr][1])
        split_particles_to_buffers[dim_grid_1d, dim_block_1d](
            particle_array, left_buffer, stay_buffer, right_buffer, i_min, i_max
        )
        # Assign the stay_buffer to the initial particle data array
        # and fill the sending buffers (if needed for MPI)
        setattr(attr_list[i_attr][0], attr_list[i_attr][1], stay_buffer)
        if left_proc is not None:
            left_buffer.get(out=uint_send_left[i_attr])
        if right_proc is not None:
            right_buffer.get(out=uint_send_right[i_attr])

    # Change the new total number of particles
    species.Ntot = new_Ntot

    # Return the sending buffers
    return (float_send_left, float_send_right, uint_send_left, uint_send_right)


def add_buffers_to_particles(
    species, float_recv_left, float_recv_right, uint_recv_left, uint_recv_right
):
    """
    Add the particles stored in recv_left and recv_right
    to the existing particle in species.

    Resize the auxiliary arrays of the particles Ex, Ey, Ez, Bx, By, Bz,
    as well as cell_idx, sorted_idx and sorting_buffer

    Parameters
    ----------
    species: a Particles object
        Contain the particles that stayed on the present processors

    float_recv_left, float_recv_right, uint_recv_left, uint_recv_right:
        arrays of shape (n_float,Nptcl) and (n_int,Nptcl) where Nptcl
        is the number of particles that are received to the left
        proc and right proc respectively, and where n_float and n_int
        are the number of float and integer quantities respectively
        These arrays are always on the CPU (since they were used for MPI)
    """
    # Copy the buffers to an enlarged array
    if species.use_cuda:
        add_buffers_gpu(
            species, float_recv_left, float_recv_right, uint_recv_left, uint_recv_right
        )
    else:
        add_buffers_cpu(
            species, float_recv_left, float_recv_right, uint_recv_left, uint_recv_right
        )

    # Reallocate the particles auxiliary arrays. This needs to be done,
    # as the total number of particles in this domain has changed.
    resize_auxiliary_arrays(species)


def resize_auxiliary_arrays(species):
    """
    Reallocate the auxiliary arrays of `species` (field-on-particle and
    sorting arrays) to its new number of particles, and mark the
    particles as unsorted.
    """
    if species.use_cuda:
        shape = (species.Ntot,)
        # Reallocate empty field-on-particle arrays on the GPU
        species.Ex = cupy.empty(shape, dtype=np.float64)
        species.Ex = cupy.empty(shape, dtype=np.float64)
        species.Ey = cupy.empty(shape, dtype=np.float64)
        species.Ez = cupy.empty(shape, dtype=np.float64)
        species.Bx = cupy.empty(shape, dtype=np.float64)
        species.By = cupy.empty(shape, dtype=np.float64)
        species.Bz = cupy.empty(shape, dtype=np.float64)
        # Reallocate empty auxiliary sorting arrays on the GPU
        species.cell_idx = cupy.empty(shape, dtype=np.int32)
        species.sorted_idx = cupy.empty(shape, dtype=np.intp)
        species.sorting_buffer = cupy.empty(shape, dtype=np.float64)
        if species.n_integer_quantities > 0:
            species.int_sorting_buffer = cupy.empty(shape, dtype=np.uint64)
    else:
        # Reallocate empty field-on-particle arrays on the CPU
        species.Ex = np.empty(species.Ntot, dtype=np.float64)
        species.Ey = np.empty(species.Ntot, dtype=np.float64)
        species.Ez = np.empty(species.Ntot, dtype=np.float64)
        species.Bx = np.empty(species.Ntot, dtype=np.float64)
        species.By = np.empty(species.Ntot, dtype=np.float64)
        species.Bz = np.empty(species.Ntot, dtype=np.float64)

    # The particles are unsorted after adding new particles.
    species.sorted = False


def add_buffers_cpu(
    species, float_recv_left, float_recv_right, uint_recv_left, uint_recv_right
):
    """
    Add the particles stored in recv_left and recv_right
    to the existing particle in species.

    Parameters
    ----------
    species: a Particles object
        Contain the particles that stayed on the present processors

    float_recv_left, float_recv_right, uint_recv_left, uint_recv_right:
        arrays of shape (n_float,Nptcl) and (n_int,Nptcl) where Nptcl
        is the number of particles that are received to the left
        proc and right proc respectively, and where n_float and n_int
        are the number of float and integer quantities respectively
        These arrays are always on the CPU (since they were used for MPI)
    """
    # Form the new particle arrays by adding the received particles
    # from the left and the right to the particles that stay in the domain
    species.x = np.hstack((float_recv_left[0], species.x, float_recv_right[0]))
    species.y = np.hstack((float_recv_left[1], species.y, float_recv_right[1]))
    species.z = np.hstack((float_recv_left[2], species.z, float_recv_right[2]))
    species.ux = np.hstack((float_recv_left[3], species.ux, float_recv_right[3]))
    species.uy = np.hstack((float_recv_left[4], species.uy, float_recv_right[4]))
    species.uz = np.hstack((float_recv_left[5], species.uz, float_recv_right[5]))
    species.inv_gamma = np.hstack(
        (float_recv_left[6], species.inv_gamma, float_recv_right[6])
    )
    species.w = np.hstack((float_recv_left[7], species.w, float_recv_right[7]))
    i_attr = 0
    if species.tracker is not None:
        species.tracker.id = np.hstack(
            (uint_recv_left[i_attr], species.tracker.id, uint_recv_right[i_attr])
        )
        i_attr += 1
    if species.ionizer is not None:
        species.ionizer.ionization_level = np.hstack(
            (
                uint_recv_left[i_attr],
                species.ionizer.ionization_level,
                uint_recv_right[i_attr],
            )
        )
        species.ionizer.w_times_level = np.hstack(
            (float_recv_left[8], species.ionizer.w_times_level, float_recv_right[8])
        )
    if species.spin_tracker is not None:
        species.spin_tracker.sx = np.hstack(
            (float_recv_left[-3], species.spin_tracker.sx, float_recv_right[-3])
        )
        species.spin_tracker.sy = np.hstack(
            (float_recv_left[-2], species.spin_tracker.sy, float_recv_right[-2])
        )
        species.spin_tracker.sz = np.hstack(
            (float_recv_left[-1], species.spin_tracker.sz, float_recv_right[-1])
        )

    # Adapt the total number of particles
    species.Ntot = species.Ntot + float_recv_left.shape[1] + float_recv_right.shape[1]


@catch_gpu_memory_error
def add_buffers_gpu(
    species, float_recv_left, float_recv_right, uint_recv_left, uint_recv_right
):
    """
    Add the particles stored in recv_left and recv_right
    to the existing particle in species.

    Parameters
    ----------
    species: a Particles object
        Contain the particles that stayed on the present processors

    float_recv_left, float_recv_right, uint_recv_left, uint_recv_right:
        arrays of shape (n_float,Nptcl) and (n_int,Nptcl) where Nptcl
        is the number of particles that are received to the left
        proc and right proc respectively, and where n_float and n_int
        are the number of float and integer quantities respectively
        These arrays are always on the CPU (since they were used for MPI)
    """
    # Get the new number of particles
    old_Ntot = species.Ntot
    n_left = float_recv_left.shape[1]
    n_right = float_recv_right.shape[1]
    new_Ntot = old_Ntot + n_left + n_right

    # Get the threads per block and the blocks per grid
    n_left_grid, n_left_block = cuda_tpb_bpg_1d(n_left)
    n_right_grid, n_right_block = cuda_tpb_bpg_1d(n_right)
    n_old_grid, n_old_block = cuda_tpb_bpg_1d(old_Ntot)

    # Iterate over particle attributes
    # Build list of float attributes to copy
    attr_list = [
        (species, "x"),
        (species, "y"),
        (species, "z"),
        (species, "ux"),
        (species, "uy"),
        (species, "uz"),
        (species, "inv_gamma"),
        (species, "w"),
    ]
    if species.ionizer is not None:
        attr_list += [(species.ionizer, "w_times_level")]
    if species.spin_tracker is not None:
        attr_list += [
            (species.spin_tracker, "sx"),
            (species.spin_tracker, "sy"),
            (species.spin_tracker, "sz"),
        ]
    # Loop through the float quantities
    for i_attr in range(len(attr_list)):
        # Copy the proper buffers to the GPU
        left_buffer = cupy.asarray(float_recv_left[i_attr])
        right_buffer = cupy.asarray(float_recv_right[i_attr])
        # Initialize the new particle array
        particle_array = cupy.empty((new_Ntot,), dtype=np.float64)
        # Merge the arrays on the GPU
        stay_buffer = getattr(attr_list[i_attr][0], attr_list[i_attr][1])
        if n_left != 0:
            copy_particles[n_left_grid, n_left_block](
                n_left, left_buffer, 0, particle_array, 0
            )
        if old_Ntot != 0:
            copy_particles[n_old_grid, n_old_block](
                old_Ntot, stay_buffer, 0, particle_array, n_left
            )
        if n_right != 0:
            copy_particles[n_right_grid, n_right_block](
                n_right, right_buffer, 0, particle_array, n_left + old_Ntot
            )
        # Assign the stay_buffer to the initial particle data array
        # and fill the sending buffers (if needed for MPI)
        setattr(attr_list[i_attr][0], attr_list[i_attr][1], particle_array)

    # Build list of integer quantities to copy
    attr_list = []
    if species.tracker is not None:
        attr_list.append((species.tracker, "id"))
    if species.ionizer is not None:
        attr_list.append((species.ionizer, "ionization_level"))
    # Loop through the integer quantities
    for i_attr in range(len(attr_list)):
        # Copy the proper buffers to the GPU
        left_buffer = cupy.asarray(uint_recv_left[i_attr])
        right_buffer = cupy.asarray(uint_recv_right[i_attr])
        # Initialize the new particle array
        particle_array = cupy.empty((new_Ntot,), dtype=np.uint64)
        # Merge the arrays on the GPU
        stay_buffer = getattr(attr_list[i_attr][0], attr_list[i_attr][1])
        if n_left != 0:
            copy_particles[n_left_grid, n_left_block](
                n_left, left_buffer, 0, particle_array, 0
            )
        if old_Ntot != 0:
            copy_particles[n_old_grid, n_old_block](
                old_Ntot, stay_buffer, 0, particle_array, n_left
            )
        if n_right != 0:
            copy_particles[n_right_grid, n_right_block](
                n_right, right_buffer, 0, particle_array, n_left + old_Ntot
            )
        # Assign the stay_buffer to the initial particle data array
        # and fill the sending buffers (if needed for MPI)
        setattr(attr_list[i_attr][0], attr_list[i_attr][1], particle_array)

    # Adapt the total number of particles
    species.Ntot = new_Ntot


def shift_particles_periodic_subdomain(species, zmin, zmax):
    """
    Assuming the local subdomain is periodic:
    Shift the particle positions by an integer number of box length,
    so that outside particle are back inside the physical domain

    Parameters:
    -----------
    species: an fbpic.Species object
        Contains the particle data
    zmin, zmax: floats
        Positions of the edges of the periodic box
    """
    # Perform the shift on the GPU
    if species.use_cuda:
        dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d(species.Ntot)
        shift_particles_periodic_cuda[dim_grid_1d, dim_block_1d](species.z, zmin, zmax)
    # Perform the shift on the CPU
    else:
        shift_particles_periodic_numba(species.z, zmin, zmax)


@numba.jit(nopython=True)
def shift_particles_periodic_numba(z, zmin, zmax):
    """
    Shift the particle positions by an integer number of box length,
    so that outside particle are back inside the physical domain

    Parameters:
    -----------
    z: 1darray of floats
        The z position of the particles (one element per particle)
    zmin, zmax: floats
        Positions of the edges of the periodic box
    """
    # Get box length
    l_box = zmax - zmin
    # Loop through the particles and shift their positions
    for i in range(len(z)):
        while z[i] >= zmax:
            z[i] -= l_box
        while z[i] < zmin:
            z[i] += l_box


# NCCL particle exchange (all species together)
# ---------------------------------------------
# The functions below keep the exchanged particles on the GPU. Each species
# packs its outgoing particles into one flat buffer per side, attribute after
# attribute (float attributes first, then integer attributes as uint64 views
# of the same 8-byte words). See NCCL_MIGRATION.md, Algorithm 7.

# Rows of the (14, n_species) int64 count array. Each exchanged block holds
# (number of particles, number of float attributes, number of integer
# attributes), so that the count exchange also checks that neighbors use
# the same buffer layout.
SEND_L, SEND_R, RECV_R, RECV_L = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12)
N_SEND_L, N_SEND_R, N_RECV_R, N_RECV_L = 0, 3, 6, 9
I_MIN, I_MAX = 12, 13
N_COUNT_ROWS = 14


def get_particle_attributes(species):
    """
    Return the exchanged attributes of `species`, in buffer order.

    Returns
    -------
    float_attrs, int_attrs: lists of (object, attribute name)
        The float64 and uint64 particle arrays (same order as the MPI path)
    """
    float_attrs = [
        (species, "x"),
        (species, "y"),
        (species, "z"),
        (species, "ux"),
        (species, "uy"),
        (species, "uz"),
        (species, "inv_gamma"),
        (species, "w"),
    ]
    if species.ionizer is not None:
        float_attrs.append((species.ionizer, "w_times_level"))
    if species.spin_tracker is not None:
        float_attrs += [
            (species.spin_tracker, "sx"),
            (species.spin_tracker, "sy"),
            (species.spin_tracker, "sz"),
        ]
    int_attrs = []
    if species.tracker is not None:
        int_attrs.append((species.tracker, "id"))
    if species.ionizer is not None:
        int_attrs.append((species.ionizer, "ionization_level"))
    return float_attrs, int_attrs


class ParticleExchangeBuffers(object):
    """
    Persistent GPU buffers for the NCCL particle exchange.

    The sending and receiving buffers are flat float64 arrays (with uint64
    views for the integer attributes). They grow when an exchange needs
    more words, and are reused otherwise.
    """

    growth_factor = 1.5

    def __init__(self):
        self.counts = None
        self.buffers = {}

    @catch_gpu_memory_error
    def get_counts(self, n_species):
        """Return the (N_COUNT_ROWS, n_species) int64 count array on the GPU."""
        if self.counts is None or self.counts.shape[1] != n_species:
            self.counts = cupy.zeros((N_COUNT_ROWS, n_species), dtype=np.int64)
        return self.counts

    @catch_gpu_memory_error
    def get(self, name, n_words, dtype):
        """
        Return the first `n_words` words of buffer `name` ('send_l',
        'send_r', 'recv_l' or 'recv_r'), as a C-contiguous array of `dtype`
        (float64 or uint64). Grow the buffer if needed.
        """
        buffer = self.buffers.get(name)
        if buffer is None or buffer.size < n_words:
            n_alloc = max(n_words, int(self.growth_factor * n_words), 1)
            buffer = cupy.empty(n_alloc, dtype=np.float64)
            self.buffers[name] = buffer
        return buffer[:n_words].view(dtype)


@catch_gpu_memory_error
def count_particles_gpu(
    species, fld, n_guard, left_proc, right_proc, i_species, counts
):
    """
    Write the kept range and the sending counts of `species` to column
    `i_species` of `counts`, on the GPU (no host synchronization).

    Parameters
    ----------
    species: a Particles object
        Contains the data of this species (sorted here if needed)

    fld: a Fields object
        Contains information about the dimension of the grid

    n_guard: int
        Number of guard cells

    left_proc, right_proc: int or None
        The neighboring processors (None for an open boundary, where the
        outgoing particles are removed and not sent)

    i_species: int
        Column of `counts` for this species

    counts: (N_COUNT_ROWS, n_species) int64 device array
        Blocks SEND_L, SEND_R and rows I_MIN, I_MAX are filled;
        blocks RECV_R, RECV_L are set to 0 (filled by the count exchange)
    """
    # Check if particles are sorted (as in `remove_particles_gpu`)
    if not species.sorted:
        species.sort_particles(fld=fld)
        species.sorted = True

    # Same index range as in `remove_particles_gpu`
    Nz = fld.Nz
    Nr = fld.Nr
    iz_min = max(n_guard + species.prefix_sum_shift, 0)
    iz_max = min(Nz - n_guard + species.prefix_sum_shift + 1, Nz)
    float_attrs, int_attrs = get_particle_attributes(species)
    count_particles_to_exchange[1, 1](
        species.prefix_sum,
        iz_min * (Nr + 1) - 1,
        iz_max * (Nr + 1) - 1,
        species.Ntot,
        left_proc is not None,
        right_proc is not None,
        len(float_attrs),
        len(int_attrs),
        i_species,
        counts,
    )


@catch_gpu_memory_error
def pack_particles_gpu(species, h_counts, i_species, send_l, send_r, start_l, start_r):
    """
    Copy the outgoing particles of `species` to the flat sending buffers.

    Parameters
    ----------
    species: a Particles object

    h_counts: (N_COUNT_ROWS, n_species) int64 host array
        The counts of all species (after the count exchange)

    i_species: int
        Column of `h_counts` for this species

    send_l, send_r: dicts of 1d device arrays
        The sending buffers, as views of dtype float64 and uint64
        (keys np.float64 and np.uint64)

    start_l, start_r: int
        First word of this species in send_l and send_r
    """
    N_l = int(h_counts[N_SEND_L, i_species])
    N_r = int(h_counts[N_SEND_R, i_species])
    i_max = int(h_counts[I_MAX, i_species])
    float_attrs, int_attrs = get_particle_attributes(species)
    dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d(N_l + N_r)
    for i_attr, (obj, name) in enumerate(float_attrs + int_attrs):
        dtype = np.float64 if i_attr < len(float_attrs) else np.uint64
        pack_particles_to_buffers[dim_grid_1d, dim_block_1d](
            getattr(obj, name),
            i_max,
            N_l,
            N_r,
            send_l[dtype],
            start_l + i_attr * N_l,
            send_r[dtype],
            start_r + i_attr * N_r,
        )


@catch_gpu_memory_error
def unpack_particles_gpu(
    species,
    h_counts,
    i_species,
    recv_l,
    recv_r,
    start_l,
    start_r,
    injected=None,
):
    """
    Build the new particle arrays of `species` on the GPU, in the order
    [received from left | kept | received from right], as `add_buffers_gpu`.
    Resize the auxiliary arrays.

    Parameters
    ----------
    species: a Particles object

    h_counts: (N_COUNT_ROWS, n_species) int64 host array
        The counts of all species (after the count exchange)

    i_species: int
        Column of `h_counts` for this species

    recv_l, recv_r: dicts of 1d device arrays
        The receiving buffers, as views of dtype float64 and uint64
        (keys np.float64 and np.uint64)

    start_l, start_r: int
        First word of this species in recv_l and recv_r

    injected: tuple (float_buffer, uint_buffer) of 2d host arrays, or None
        Particles from continuous injection. They replace the particles
        received from the right (as in the MPI path).
    """
    M_l = int(h_counts[N_RECV_L, i_species])
    M_r = int(h_counts[N_RECV_R, i_species])
    i_min = int(h_counts[I_MIN, i_species])
    N_stay = int(h_counts[I_MAX, i_species]) - i_min
    if injected is not None:
        M_r = injected[0].shape[1]
    new_Ntot = M_l + N_stay + M_r

    float_attrs, int_attrs = get_particle_attributes(species)
    dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d(new_Ntot)
    for i_attr, (obj, name) in enumerate(float_attrs + int_attrs):
        is_float = i_attr < len(float_attrs)
        dtype = np.float64 if is_float else np.uint64
        if injected is not None:
            # One row of the injected particles, uploaded to the GPU
            row = (
                injected[0][i_attr]
                if is_float
                else injected[1][i_attr - len(float_attrs)]
            )
            right_buffer = cupy.asarray(row)
            right_start = 0
        else:
            right_buffer = recv_r[dtype]
            right_start = start_r + i_attr * M_r
        particle_array = cupy.empty((new_Ntot,), dtype=dtype)
        unpack_particles_from_buffers[dim_grid_1d, dim_block_1d](
            particle_array,
            getattr(obj, name),
            i_min,
            N_stay,
            recv_l[dtype],
            start_l + i_attr * M_l,
            M_l,
            right_buffer,
            right_start,
            M_r,
        )
        setattr(obj, name, particle_array)

    species.Ntot = new_Ntot
    resize_auxiliary_arrays(species)
    return M_l, N_stay, M_r


# Cuda routines
# -------------
if cuda_installed:

    @warp_kernel
    def split_particles_to_buffers(
        particle_array: wp.array(dtype=Any),
        left_buffer: wp.array(dtype=Any),
        stay_buffer: wp.array(dtype=Any),
        right_buffer: wp.array(dtype=Any),
        i_min: int,
        i_max: int,
    ):
        """
        Split the (sorted) particle array into the three arrays left_buffer,
        stay_buffer and right_buffer (in the same order)

        Parameters:
        ------------
        particle_array: 1d device arrays of floats
            Original array of particles
            (represents *one* of the particle quantities)

        left_buffer, right_buffer: 1d device arrays of floats
            Will contain the particles that are outside of the physical domain
            Note: if the boundary is open, then these buffers have size 0
            and in this case, they will not be filled
            (the corresponding particles are simply lost)

        stay_buffer: 1d device array of floats
            Will contain the particles that are inside the physical domain

        i_min, i_max: int
            Indices of particle_array between which particles are kept
            (and thus copied to stay_buffer). The particles below i_min
            (resp. above i_max) are copied to left_buffer (resp. right_buffer)
        """
        # Get a 1D CUDA grid (the index corresponds to a particle index)
        i = wp.tid()

        # Auxiliary variables
        n_left = left_buffer.shape[0]
        n_right = right_buffer.shape[0]
        Ntot = particle_array.shape[0]

        # Copy the particles into the right buffer
        if i < i_min:
            # Check whether buffer is not empty (open boundary)
            if n_left != 0:
                left_buffer[i] = particle_array[i]
        elif i < i_max:
            stay_buffer[i - i_min] = particle_array[i]
        elif i < Ntot:
            # Check whether buffer is not empty (open boundary)
            if n_right != 0:
                right_buffer[i - i_max] = particle_array[i]

    @warp_kernel
    def copy_particles(
        N_elements: int,
        source_array: wp.array(dtype=Any),
        source_start: int,
        target_array: wp.array(dtype=Any),
        target_start: int,
    ):
        """
        Copy `N_elements` elements from `source_array` to `target_array`

        Parameters:
        ------------
        N_elements: int
            The number of elements to copy

        source_array, target_array: 1d device arrays of floats
            The arrays from/to which the data should be copied
            (represents *one* of the particle quantities)

        source_start, target_start: ints
            The indices at which to start the copy, in both the
            source and the target.
        """
        # Get a 1D CUDA grid (the index corresponds to a particle index)
        i = wp.tid()

        # Copy the particles into the right buffer
        if i < N_elements:
            target_array[i + target_start] = source_array[i + source_start]

    @warp_kernel
    def shift_particles_periodic_cuda(
        z: wp.array(dtype=wp.float64),
        zmin: wp.float64,
        zmax: wp.float64,
    ):
        """
        Shift the particle positions by an integer number of box length,
        so that outside particle are back inside the physical domain

        Parameters:
        -----------
        z: 1darray of floats
            The z position of the particles (one element per particle)
        zmin, zmax: floats
            Positions of the edges of the periodic box
        """
        # Get a 1D CUDA grid (the index corresponds to a particle index)
        i = wp.tid()
        # Get box length
        l_box = zmax - zmin
        # Shift particle position
        if i < z.shape[0]:
            while z[i] >= zmax:
                z[i] -= l_box
            while z[i] < zmin:
                z[i] += l_box

    @warp_kernel
    def count_particles_to_exchange(
        prefix_sum: wp.array(dtype=wp.int32),
        i_cell_min: int,
        i_cell_max: int,
        Ntot: int,
        send_left: bool,
        send_right: bool,
        n_float: int,
        n_int: int,
        i_species: int,
        counts: wp.array2d(dtype=wp.int64),
    ):
        """
        Read the kept particle range of one species from its prefix sum,
        and write it to column `i_species` of `counts` (launch with 1 thread).

        Parameters:
        ------------
        prefix_sum: 1d device array of ints
            Inclusive prefix sum of the number of particles per cell

        i_cell_min, i_cell_max: int
            Last cell before the kept cells (-1 if none), and last kept cell

        Ntot: int
            Number of particles of the species

        send_left, send_right: bool
            Whether the particles outside the kept range are sent to the
            left and right neighbors (False at an open boundary)

        n_float, n_int: int
            Number of float and integer attributes of the species

        i_species: int
            Column of `counts`

        counts: 2d device array of int64, shape (N_COUNT_ROWS, n_species)
            Rows 0-2: (number sent left, n_float, n_int);
            rows 3-5: (number sent right, n_float, n_int);
            rows 6-11: received from the right and from the left (0 here);
            row 12: first kept index (i_min); row 13: end of the kept range
        """
        if wp.tid() == 0:
            i_min = wp.int64(0)
            if i_cell_min >= 0:
                i_min = wp.int64(prefix_sum[i_cell_min])
            i_max = wp.int64(prefix_sum[i_cell_max])
            N_left = wp.int64(0)
            if send_left:
                N_left = i_min
            N_right = wp.int64(0)
            if send_right:
                N_right = wp.int64(Ntot) - i_max
            counts[0, i_species] = N_left
            counts[1, i_species] = wp.int64(n_float)
            counts[2, i_species] = wp.int64(n_int)
            counts[3, i_species] = N_right
            counts[4, i_species] = wp.int64(n_float)
            counts[5, i_species] = wp.int64(n_int)
            for row in range(6, 12):
                counts[row, i_species] = wp.int64(0)
            counts[12, i_species] = i_min
            counts[13, i_species] = i_max

    @warp_kernel
    def pack_particles_to_buffers(
        particle_array: wp.array(dtype=Any),
        i_max: int,
        N_left: int,
        N_right: int,
        left_buffer: wp.array(dtype=Any),
        left_start: int,
        right_buffer: wp.array(dtype=Any),
        right_start: int,
    ):
        """
        Copy the first `N_left` particles to `left_buffer` (from index
        `left_start`) and the `N_right` particles from `i_max` to
        `right_buffer` (from index `right_start`). Launch N_left + N_right
        threads.

        Parameters:
        ------------
        particle_array: 1d device array
            One of the (sorted) particle quantities

        i_max: int
            First particle sent to the right

        N_left, N_right: int
            Number of particles sent to the left and to the right

        left_buffer, right_buffer: 1d device arrays (same dtype)
            Flat sending buffers

        left_start, right_start: int
            Offsets of this quantity in the sending buffers
        """
        i = wp.tid()
        if i < N_left:
            left_buffer[left_start + i] = particle_array[i]
        elif i < N_left + N_right:
            right_buffer[right_start + i - N_left] = particle_array[i_max + i - N_left]

    @warp_kernel
    def unpack_particles_from_buffers(
        new_array: wp.array(dtype=Any),
        old_array: wp.array(dtype=Any),
        i_min: int,
        N_stay: int,
        left_buffer: wp.array(dtype=Any),
        left_start: int,
        N_left: int,
        right_buffer: wp.array(dtype=Any),
        right_start: int,
        N_right: int,
    ):
        """
        Fill `new_array` with [N_left received from the left | N_stay kept
        particles from `old_array`, from i_min | N_right received from the
        right]. Launch N_left + N_stay + N_right threads.

        Parameters:
        ------------
        new_array: 1d device array
            The new particle quantity

        old_array: 1d device array
            The (sorted) particle quantity before the exchange

        i_min, N_stay: int
            First kept particle in `old_array`, and number of kept particles

        left_buffer, right_buffer: 1d device arrays (same dtype)
            Receiving buffers (or injected particles, for right_buffer)

        left_start, right_start: int
            Offsets of this quantity in the receiving buffers

        N_left, N_right: int
            Number of particles received from the left and from the right
        """
        j = wp.tid()
        if j < N_left:
            new_array[j] = left_buffer[left_start + j]
        elif j < N_left + N_stay:
            new_array[j] = old_array[i_min + j - N_left]
        elif j < N_left + N_stay + N_right:
            new_array[j] = right_buffer[right_start + j - N_left - N_stay]
