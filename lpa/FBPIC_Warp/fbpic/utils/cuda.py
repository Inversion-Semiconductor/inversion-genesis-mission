# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of generic functions that operate on a GPU.
"""
import warnings
import numpy as np

# Both libraries must see a CUDA device. CPU execution remains available otherwise.
try:
    import cupy

    cupy_installed = cupy.is_available()
except (ImportError, RuntimeError, AssertionError):
    cupy_installed = False

try:
    import warp as wp

    warp_cuda_installed = wp.is_cuda_available() if cupy_installed else False
except ImportError:
    warp_cuda_installed = False

cuda_installed = cupy_installed and warp_cuda_installed
cuda_gpu_model = "other"
if cuda_installed:
    gpu_name = cupy.cuda.runtime.getDeviceProperties(0)["name"]
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    if "P100" in gpu_name:
        cuda_gpu_model = "P100"
    elif "V100" in gpu_name or "A100" in gpu_name:
        # Upstream FBPIC also gives A100 the V100 launch settings.
        cuda_gpu_model = "V100"

try:
    import pynvml

    pynvml_installed = True
except ImportError:
    pynvml_installed = False

# -----------------------------------------------------
# CUDA grid utilities
# -----------------------------------------------------


def cuda_tpb_bpg_1d(x, TPB=256):
    """
    Get the needed blocks per grid for a 1D CUDA grid.

    Parameters :
    ------------
    x : int
        Total number of threads

    TPB : int
        Threads per block

    Returns :
    ---------
    BPG : int
        Number of blocks per grid

    TPB : int
        Threads per block.
    """
    # Calculates the needed blocks per grid
    BPG = int(x / TPB + 1)
    return BPG, TPB


def cuda_tpb_bpg_2d(x, y, TPBx=1, TPBy=128):
    """
    Get the needed blocks per grid for a 2D CUDA grid.

    Parameters :
    ------------
    x, y  : int
        Total number of threads in first and second dimension

    TPBx, TPBy : int
        Threads per block in x and y

    Returns :
    ------------
    (BPGx, BPGy) : tuple of ints
        Number of blocks per grid in x and y

    (TPBx, TPBy) : tuple of ints
        Threads per block in x and y.
    """
    # Calculates the needed blocks per grid
    BPGx = int(x / TPBx + 1)
    BPGy = int(y / TPBy + 1)
    return (BPGx, BPGy), (TPBx, TPBy)


# -----------------------------------------------------
# CUDA memory management
# -----------------------------------------------------


def send_data_to_gpu(simulation):
    """
    Send the simulation data to the GPU.
    Calls the functions of the particle and field package
    that send the data to the GPU.

    Parameters :
    ------------
    simulation : object
        A simulation object that contains the particle
        (ptcl) and field object (fld)
    """
    # Send particles to the GPU (if CUDA is used)
    for species in simulation.ptcl:
        if species.use_cuda:
            species.send_particles_to_gpu()
    # Send fields to the GPU (if CUDA is used)
    simulation.fld.send_fields_to_gpu()


def receive_data_from_gpu(simulation):
    """
    Receive the simulation data from the GPU.
    Calls the functions of the particle and field package
    that receive the data from the GPU.

    Parameters :
    ------------
    simulation : object
        A simulation object that contains the particle
        (ptcl) and field object (fld)
    """
    # Receive the particles from the GPU (if CUDA is used)
    for species in simulation.ptcl:
        if species.use_cuda:
            species.receive_particles_from_gpu()
    # Receive fields from the GPU (if CUDA is used)
    simulation.fld.receive_fields_from_gpu()


class GpuMemoryManager(object):
    """
    Context manager that temporarily moves the simulation data to the GPU,
    if the data is originally on the CPU when entering the context manager
    """

    def __init__(self, simulation):
        """
        Initialize the context manager

        Parameters:
        -----------
        simulation: object
            A simulation object that contains the particle
            (ptcl) and field object (fld)
        """
        # Check whether the data is initially on the CPU or GPU
        self.fields_were_on_gpu = simulation.fld.data_is_on_gpu
        self.species_were_on_gpu = [
            species.data_is_on_gpu for species in simulation.ptcl
        ]
        # Keep a reference to the simulation
        self.sim = simulation

    def __enter__(self):
        """
        Move the data to the GPU (if it was originally on the CPU)
        """
        if self.sim.use_cuda:
            if not self.fields_were_on_gpu:
                self.sim.fld.send_fields_to_gpu()
            for i, species in enumerate(self.sim.ptcl):
                if not self.species_were_on_gpu[i]:
                    species.send_particles_to_gpu()

    def __exit__(self, type, value, traceback):
        """
        Move the data back to the CPU (if it was originally on the CPU)
        """
        if self.sim.use_cuda:
            if not self.fields_were_on_gpu:
                self.sim.fld.receive_fields_from_gpu()
            for i, species in enumerate(self.sim.ptcl):
                if not self.species_were_on_gpu[i]:
                    species.receive_particles_from_gpu()


# -----------------------------------------------------
# CUDA mpi management
# -----------------------------------------------------


def get_uuid(gpu_id):
    """
    Returns the UUID of a GPU device (or None if it cannot determine it)

    Parameters:
    -----------
    gpu_id: Local device id of the GPU (int)

    Returns:
    --------
    uuid: Unique identifier (UUID) of the GPU (str)
    """
    # Get UUID using cupy
    uuid = cupy.cuda.runtime.getDeviceProperties(gpu_id)["uuid"]

    # Check the UUID length to prevent crashes
    if len(uuid) != 16:
        warnings.warn(
            f"Failed to detect UUID of GPU {gpu_id} (invalid UUID length: {len(uuid)})"
        )
        return None

    # conversion strategy from numba PR #6700
    b = "%02x"
    b2 = b * 2
    b4 = b * 4
    b6 = b * 6
    fmt = f"GPU-{b4}-{b2}-{b2}-{b2}-{b6}"
    return fmt % tuple(bytes(uuid))


def get_uuid_alt(gpu_id):
    """
    Returns the UUID of a GPU device using `pynvml`.

    Parameters:
    -----------
    gpu_id: Local device id of the GPU (int)

    Returns:
    --------
    uuid: Unique identifier (UUID) of the GPU (str)
    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    uuid = pynvml.nvmlDeviceGetUUID(handle)
    pynvml.nvmlShutdown()
    return uuid


def check_consecutive_ranks_on_same_nodes(mpi):
    """
    Check that consecutive MPI ranks are on the same nodes, and
    return a corresponding boolean. Print a warning message if
    it is not the case, this this can affect performance.
    """
    # Skip this function if there is only one MPI rank
    if mpi.COMM_WORLD.size == 1:
        return True

    # Check that the decomposition is standard, raise a warning otherwise
    nodes = mpi.COMM_WORLD.gather(mpi.Get_processor_name(), root=0)
    standard_decomp = None
    if mpi.COMM_WORLD.rank == 0:
        # Check that consecutive MPI ranks are on the same node
        standard_decomp = True
        unique_nodes = [nodes[0]]  # Initialize list of unique nodes
        for i in range(1, len(nodes)):
            if nodes[i] != nodes[i - 1]:  # Consecutive ranks not on the same node
                if nodes[i] not in unique_nodes:
                    unique_nodes.append(nodes[i])
                else:
                    # This node was seen before ; this means that several
                    # MPI ranks selected it, but that they are not consecutive.
                    standard_decomp = False
                    break
        # Print a corresponding warning
        if not standard_decomp:
            warnings.warn(
                "It seems that the distribution of MPI ranks on compute nodes is such\n"
                "that consecutive MPI ranks are not located on the same node.\n"
                "(See the FBPIC output with `verbose_level = 2` for more details.)\n"
                "This type of MPI distribution can degrade the performance of FBPIC.\n"
                "Please check the options of your MPI launcher (e.g. `mpirun`, `srun`)\n"
                "in order to use a different MPI distribution."
            )
    standard_decomp = mpi.COMM_WORLD.bcast(standard_decomp, root=0)
    return standard_decomp


def mpi_select_gpus(mpi):
    """
    Selects the correct GPU used by the current MPI process
    using the MPI rank

    Parameters :
    ------------
    mpi: an mpi4py.MPI object
    """
    std_decomp = check_consecutive_ranks_on_same_nodes(mpi)
    if not std_decomp:
        # Convert node name to integer, and create a local communicator
        # for MPI ranks that are on the same node
        node_name = mpi.Get_processor_name()
        color = int.from_bytes(node_name.encode(), "little") % 100000000
        comm = mpi.COMM_WORLD.Split(color=color)
    else:
        comm = mpi.COMM_WORLD

    # Attribute a GPU to each rank
    n_gpus = cupy.cuda.runtime.getDeviceCount()
    rank = comm.rank
    for i_gpu in range(n_gpus):
        if rank % n_gpus == i_gpu:
            # CuPy and Warp share one explicit stream on the selected device.
            from .warp import get_runtime

            get_runtime(i_gpu).use()
            if pynvml_installed:
                uuid = get_uuid_alt(i_gpu)
            else:
                uuid = get_uuid(i_gpu)
        mpi.COMM_WORLD.barrier()

    # Gather unique GPU identifiers
    uuids = mpi.COMM_WORLD.gather(uuid)

    # Check that no GPU was selected more than once
    if rank == 0:
        if None not in uuids and (len(uuids) > len(set(uuids))):
            warnings.warn(
                "GPUs have been oversubscribed by MPI ranks.\n"
                "This means that the same GPU was selected by more than one "
                "parallel process,\nwhich will result in poor performance.\n"
                "(See the FBPIC output with `verbose_level = 2` for more details.)"
            )


def pinned_array(shape: int | tuple[int, ...], dtype=np.float64) -> np.ndarray:
    """Allocate a NumPy array in CuPy-owned page-locked host memory.

    Args:
        shape: Array dimensions.
        dtype: NumPy element dtype.

    Returns:
        A writable host array for asynchronous CUDA transfers.
    """
    import cupyx

    return cupyx.empty_pinned(shape, dtype=dtype)


class PinnedStagingBuffer:
    """Reuse one page-locked host buffer for repeated host-to-device copies.

    The capacity only grows, with headroom, so that repeated copies of similar
    sizes do not allocate page-locked memory again. The buffer is allocated
    directly with `cudaHostAlloc`, not from the CuPy pinned memory pool.
    Before `array` returns the buffer again, it waits for the previous copy
    from the buffer, because that copy is asynchronous.
    """

    def __init__(self, headroom: float = 1.25) -> None:
        """Create an empty staging buffer.

        Args:
            headroom: Ratio of the allocated capacity to the requested size,
                when the buffer grows.
        """
        self.headroom = headroom
        self._memory = None
        self._capacity = 0
        self._event = None

    def array(self, shape: tuple[int, ...], dtype=np.float64) -> np.ndarray:
        """Return a writable, C-contiguous host array in the staging buffer.

        The array shares memory with the arrays that earlier calls returned.
        This call blocks until the previous `to_device` copy has completed.

        Args:
            shape: Array dimensions.
            dtype: NumPy element dtype.

        Returns:
            A host array in page-locked memory, or in regular memory when the
            array is empty.
        """
        if self._event is not None:
            self._event.synchronize()
            self._event = None
        count = int(np.prod(shape))
        nbytes = count * np.dtype(dtype).itemsize
        if nbytes == 0:
            return np.empty(shape, dtype=dtype)
        if nbytes > self._capacity:
            # Release the old buffer before the new one is allocated
            self._memory = None
            self._capacity = int(nbytes * self.headroom)
            self._memory = cupy.cuda.PinnedMemory(self._capacity)
        pointer = cupy.cuda.PinnedMemoryPointer(self._memory, 0)
        return np.frombuffer(pointer, dtype=dtype, count=count).reshape(shape)

    def to_device(self, host_array: np.ndarray) -> "cupy.ndarray":
        """Copy an array from `array` to a new device array, asynchronously.

        The copy runs on the current CuPy stream, which is the shared runtime
        stream. The host must not modify `host_array` until the next call to
        `array`, which waits for the copy.

        Args:
            host_array: An array that `array` returned.

        Returns:
            A device array with the same shape and dtype.
        """
        stream = cupy.cuda.get_current_stream()
        device_array = cupy.empty(host_array.shape, dtype=host_array.dtype)
        device_array.set(host_array, stream=stream)
        self._event = stream.record()
        return device_array
