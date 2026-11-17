
def gpu_occupancy(
    threads_per_block: int, registers_per_thread: int, shared_mem_per_block: int,
    max_threads_per_sm: int, max_warps_per_sm: int, max_blocks_per_sm: int,
    max_registers_per_sm: int, max_shared_mem_per_sm: int, warp_size: int = 32,
) -> dict:
    # Number of allocated warps per block
    warps_per_block = (threads_per_block + warp_size - 1) // warp_size

    # Effective threads include partially filled warp capacity
    effective_threads = warps_per_block * warp_size

    # Resource consumption per block
    registers_per_block = registers_per_thread * effective_threads

    # Calculate limits imposed by each resource
    limits = [
        max_threads_per_sm // effective_threads,
        max_warps_per_sm // warps_per_block,
        max_blocks_per_sm,
    ]

    # Zero resource consumption does not constrain block count
    if registers_per_block > 0:
        limits.append(max_registers_per_sm // registers_per_block)

    if shared_mem_per_block > 0:
        limits.append(max_shared_mem_per_sm // shared_mem_per_block)

    # Maximum resident blocks
    blocks_per_sm = int(min(limits))

    # Occupancy calculations
    resident_warps = int(blocks_per_sm * warps_per_block)
    occupancy = float(resident_warps / max_warps_per_sm)

    return {
        "blocks_per_sm": blocks_per_sm,
        "resident_warps": resident_warps,
        "occupancy": occupancy,
    }
