from itertools import product

import numpy as onp
import jax.numpy as jnp
from jax import lax, vmap


def integer_digits(number, base, length, dtype=jnp.int32):
    """Fixed-width base-k digits, most-significant digit first."""
    powers = jnp.power(
        jnp.asarray(base, dtype=dtype),
        jnp.arange(length - 1, -1, -1, dtype=dtype),
    )
    number = jnp.asarray(number, dtype=dtype)
    return jnp.mod(jnp.floor_divide(number, powers), base).astype(dtype)


def integer_digits_array(numbers, base, length, dtype=jnp.int32):
    """Fixed-width base-k digits for an array of integers."""
    powers = jnp.power(
        jnp.asarray(base, dtype=dtype),
        jnp.arange(length - 1, -1, -1, dtype=dtype),
    )
    numbers = jnp.asarray(numbers, dtype=dtype)
    return jnp.mod(jnp.floor_divide(numbers[..., jnp.newaxis], powers), base).astype(dtype)


def wolfram_rule_table(rule_number, colors, table_size, dtype=jnp.int32):
    """Return Wolfram/NKS digit table for a rule number."""
    return integer_digits(rule_number, colors, table_size, dtype=dtype)


def table_from_outputs(outputs_by_code, dtype=jnp.int32):
    """Build a Wolfram-ordered table from direct code -> output values."""
    return jnp.asarray(outputs_by_code, dtype=dtype)[::-1]


def lookup_wolfram_table(table, codes):
    """Lookup using Wolfram's descending code order."""
    table = jnp.asarray(table)
    return jnp.take(table, table.shape[0] - 1 - codes.astype(jnp.int32), axis=0)


def lookup_wolfram_tables(tables, codes):
    """Batched Wolfram lookup for one table per leading state axis."""
    tables = jnp.asarray(tables)
    codes = jnp.asarray(codes, dtype=jnp.int32)
    if tables.ndim != 2:
        raise ValueError("tables must have shape (batch, table_size)")
    if codes.ndim < 1 or codes.shape[0] != tables.shape[0]:
        raise ValueError("codes must have a leading batch dimension matching tables")

    indices = (tables.shape[1] - 1 - codes).reshape((codes.shape[0], -1))
    values = jnp.take_along_axis(tables, indices, axis=1)
    return values.reshape(codes.shape)


def dense_offsets(ndim, radius):
    ranges = [range(-radius, radius + 1) for _ in range(ndim)]
    return tuple(tuple(offset) for offset in product(*ranges))


def moore_offsets(ndim, radius=1):
    return dense_offsets(ndim, radius)


def von_neumann_offsets(ndim, radius=1):
    offsets = dense_offsets(ndim, radius)
    return tuple(offset for offset in offsets if sum(abs(x) for x in offset) <= radius)


def offsets_1d(radius):
    return tuple((i,) for i in range(-radius, radius + 1))


def general_weights(colors, offsets, dtype=jnp.int32):
    """Positional base-k weights in offset order."""
    n = len(offsets)
    return jnp.power(
        jnp.asarray(colors, dtype=dtype),
        jnp.arange(n - 1, -1, -1, dtype=dtype),
    )


def totalistic_weights(offsets, dtype=jnp.int32):
    return jnp.ones((len(offsets),), dtype=dtype)


def outer_totalistic_weights(colors, offsets, dtype=jnp.int32):
    center = tuple(0 for _ in offsets[0])
    return jnp.asarray(
        [1 if tuple(offset) == center else colors for offset in offsets],
        dtype=dtype,
    )


def table_size_general(colors, offsets):
    return int(colors ** len(offsets))


def table_size_totalistic(colors, offsets):
    return int((colors - 1) * len(offsets) + 1)


def table_size_outer_totalistic(colors, offsets):
    center = tuple(0 for _ in offsets[0])
    outer_count = sum(1 for offset in offsets if tuple(offset) != center)
    return int(colors * (outer_count * (colors - 1) + 1))


def table_size_weighted(colors, weights):
    return int((colors - 1) * sum(weights) + 1)


def encode_shift_accumulate(state, offsets, weights):
    """Encode each cell's neighborhood by rolling and accumulating weights."""
    state = jnp.asarray(state, dtype=jnp.int32)
    weights = jnp.asarray(weights, dtype=jnp.int32)
    code = jnp.zeros_like(state, dtype=jnp.int32)
    for offset, weight in zip(offsets, weights):
        shift = tuple(-x for x in offset)
        code = code + weight * jnp.roll(state, shift=shift, axis=tuple(range(state.ndim)))
    return code


def kernel_from_offsets(offsets, weights):
    """Create a dense convolution/correlation kernel from offset weights."""
    offsets_arr = onp.asarray(offsets, dtype=onp.int64)
    pad = onp.max(onp.abs(offsets_arr), axis=0)
    shape = tuple((2 * pad + 1).tolist())
    kernel = onp.zeros(shape, dtype=onp.int32)
    for offset, weight in zip(offsets, onp.asarray(weights, dtype=onp.int32)):
        kernel[tuple((onp.asarray(offset) + pad).tolist())] = weight
    return jnp.asarray(kernel, dtype=jnp.int32)


def _dimension_chars(ndim):
    if ndim == 1:
        return "W"
    if ndim == 2:
        return "HW"
    if ndim == 3:
        return "DHW"
    chars = "ABDEFGHJKLMPQRSTUVWXYZ"
    if ndim > len(chars):
        raise ValueError(f"Too many dimensions for convolution: {ndim}")
    return chars[:ndim]


def encode_wrapped_convolution(state, kernel):
    """Encode neighborhoods with wrapped padding plus an ND convolution."""
    state = jnp.asarray(state, dtype=jnp.int32)
    kernel = jnp.asarray(kernel, dtype=jnp.int32)
    if state.ndim != kernel.ndim:
        raise ValueError("state and kernel must have the same number of dimensions")

    pad = tuple((s // 2, s // 2) for s in kernel.shape)
    padded = jnp.pad(state, pad, mode="wrap")

    spatial = _dimension_chars(state.ndim)
    lhs_spec = "N" + spatial + "C"
    rhs_spec = spatial + "IO"
    # CUDA cuDNN does not support the s32 convolution we want for CA encodings.
    # CA neighborhood sums are small in the practical Wolfram-style cases we run,
    # so float32 accumulation is exact and can be cast back before table lookup.
    lhs = padded[jnp.newaxis, ..., jnp.newaxis].astype(jnp.float32)
    rhs = kernel[..., jnp.newaxis, jnp.newaxis].astype(jnp.float32)
    result = lax.conv_general_dilated(
        lhs,
        rhs,
        window_strides=(1,) * state.ndim,
        padding="VALID",
        dimension_numbers=(lhs_spec, rhs_spec, lhs_spec),
    )
    return jnp.rint(result[0, ..., 0]).astype(jnp.int32)


def step_shift_accumulate(state, table, offsets, weights):
    codes = encode_shift_accumulate(state, offsets, weights)
    return lookup_wolfram_table(table, codes)


def step_wrapped_convolution(state, table, kernel):
    codes = encode_wrapped_convolution(state, kernel)
    return lookup_wolfram_table(table, codes)


def step_shift_accumulate_vmap(states, tables, offsets, weights):
    """Apply many 1D/ND rules in parallel, one table per state."""
    return vmap(
        lambda state, table: step_shift_accumulate(state, table, offsets, weights),
        in_axes=(0, 0),
    )(states, tables)


def step_shift_accumulate_same_state_vmap(state, tables, offsets, weights):
    """Apply many rules in parallel to the same initial state."""
    return vmap(
        lambda table: step_shift_accumulate(state, table, offsets, weights),
        in_axes=0,
    )(tables)


def step_wrapped_convolution_vmap(states, tables, kernel):
    """Apply many convolutional rules in parallel, one table per state."""
    return vmap(
        lambda state, table: step_wrapped_convolution(state, table, kernel),
        in_axes=(0, 0),
    )(states, tables)


def step_wrapped_convolution_same_state_vmap(state, tables, kernel):
    """Apply many convolutional rules in parallel to the same state."""
    return vmap(
        lambda table: step_wrapped_convolution(state, table, kernel),
        in_axes=0,
    )(tables)


def reversible_second_order_step(previous, current, local_step_fn, colors):
    """Fredkin-style reversible update: next = local(current) - previous mod k."""
    previous = jnp.asarray(previous, dtype=jnp.int32)
    current = jnp.asarray(current, dtype=jnp.int32)
    return jnp.mod(local_step_fn(current) - previous, colors).astype(jnp.int32)


def reversible_second_order_reverse_step(current, next_state, local_step_fn, colors):
    """Recover the previous state from current and next for a second-order rule."""
    current = jnp.asarray(current, dtype=jnp.int32)
    next_state = jnp.asarray(next_state, dtype=jnp.int32)
    return jnp.mod(local_step_fn(current) - next_state, colors).astype(jnp.int32)


def reversible_second_order_evolve(local_step_fn, previous, current, steps, colors):
    """Return previous, current, then steps future states for a reversible rule."""
    previous = jnp.asarray(previous, dtype=jnp.int32)
    current = jnp.asarray(current, dtype=jnp.int32)

    def scan_step(carry, _):
        prev, cur = carry
        nxt = reversible_second_order_step(prev, cur, local_step_fn, colors)
        return (cur, nxt), nxt

    _, future = lax.scan(scan_step, (previous, current), None, steps)
    return jnp.concatenate(
        [jnp.stack([previous, current]), future],
        axis=0,
    )


def is_permutation(values):
    values = list(values)
    return sorted(values) == list(range(len(values)))


def inverse_permutation(values):
    if not is_permutation(values):
        raise ValueError("values must be a permutation of range(len(values))")
    inverse = [0] * len(values)
    for i, value in enumerate(values):
        inverse[value] = i
    return tuple(inverse)


def encode_blocks(state, colors, block_shape, phase=None):
    """Encode non-overlapping phased blocks as base-k integers."""
    state = jnp.asarray(state, dtype=jnp.int32)
    block_shape = tuple(block_shape)
    if phase is None:
        phase = (0,) * state.ndim
    phase = tuple(phase)
    if state.ndim != len(block_shape) or state.ndim != len(phase):
        raise ValueError("state, block_shape, and phase must have matching dimensions")
    if any(size % block != 0 for size, block in zip(state.shape, block_shape)):
        raise ValueError("each state dimension must be divisible by its block size")

    shifted = jnp.roll(state, shift=tuple(-p for p in phase), axis=tuple(range(state.ndim)))
    grid_shape = tuple(size // block for size, block in zip(state.shape, block_shape))
    interleaved_shape = tuple(x for pair in zip(grid_shape, block_shape) for x in pair)
    interleaved = shifted.reshape(interleaved_shape)
    grid_axes = tuple(range(0, 2 * state.ndim, 2))
    block_axes = tuple(range(1, 2 * state.ndim, 2))
    blocks = jnp.transpose(interleaved, grid_axes + block_axes)
    block_cell_count = int(onp.prod(block_shape))
    flat_blocks = blocks.reshape(grid_shape + (block_cell_count,))
    weights = general_weights(colors, tuple((i,) for i in range(block_cell_count)))
    return jnp.sum(flat_blocks * weights, axis=-1).astype(jnp.int32)


def decode_blocks(codes, colors, block_shape):
    """Decode base-k block integers into block-shaped cell arrays."""
    block_shape = tuple(block_shape)
    block_cell_count = int(onp.prod(block_shape))
    digits = integer_digits_array(codes, colors, block_cell_count)
    return digits.reshape(tuple(codes.shape) + block_shape)


def apply_block_table(state, block_table, colors, block_shape, phase=None):
    """Apply a block lookup table to a phased non-overlapping partition."""
    state = jnp.asarray(state, dtype=jnp.int32)
    block_shape = tuple(block_shape)
    if phase is None:
        phase = (0,) * state.ndim
    phase = tuple(phase)

    codes = encode_blocks(state, colors, block_shape, phase)
    output_codes = jnp.take(jnp.asarray(block_table, dtype=jnp.int32), codes)
    output_blocks = decode_blocks(output_codes, colors, block_shape)

    grid_shape = codes.shape
    grid_axes = tuple(range(state.ndim))
    block_axes = tuple(range(state.ndim, 2 * state.ndim))
    interleaved_axes = tuple(
        axis
        for pair in zip(grid_axes, block_axes)
        for axis in pair
    )
    interleaved = jnp.transpose(output_blocks, interleaved_axes)
    shifted = interleaved.reshape(state.shape)
    return jnp.roll(shifted, shift=phase, axis=tuple(range(state.ndim))).astype(jnp.int32)


def block_evolve(block_table, state, colors, block_shape, phases, steps):
    """Evolve by cycling through phased block partitions."""
    phases = tuple(tuple(phase) for phase in phases)
    current = jnp.asarray(state, dtype=jnp.int32)
    history = [current]
    for i in range(steps):
        current = apply_block_table(
            current,
            block_table,
            colors,
            block_shape,
            phases[i % len(phases)],
        )
        history.append(current)
    return jnp.stack(history)


def block_run(block_table, state, colors, block_shape, phases, steps):
    """Run a phased block CA for a dynamic number of steps and return final state."""
    phases = tuple(tuple(phase) for phase in phases)
    branch_fns = tuple(
        lambda current, phase=phase: apply_block_table(
            current,
            block_table,
            colors,
            block_shape,
            phase,
        )
        for phase in phases
    )

    def body(i, current):
        return lax.switch(jnp.mod(i, len(phases)), branch_fns, current)

    return lax.fori_loop(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(steps, dtype=jnp.int32),
        body,
        jnp.asarray(state, dtype=jnp.int32),
    )


def evolve(step_fn, state, steps):
    def scan_step(current, _):
        nxt = step_fn(current)
        return nxt, nxt

    _, history = lax.scan(scan_step, jnp.asarray(state, dtype=jnp.int32), None, steps)
    return jnp.concatenate([jnp.expand_dims(state, 0), history], axis=0)


def elementary_offsets():
    return offsets_1d(1)


def elementary_weights(dtype=jnp.int32):
    return general_weights(2, elementary_offsets(), dtype=dtype)


def elementary_rule_table(rule_number):
    return wolfram_rule_table(rule_number, 2, 8)


def elementary_step(rule_number, state):
    return step_shift_accumulate(
        state,
        elementary_rule_table(rule_number),
        elementary_offsets(),
        elementary_weights(),
    )
