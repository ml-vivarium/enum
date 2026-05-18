import argparse
import time

import jax
import jax.numpy as jnp

import ca_jax as ca


def _format_shape(shape):
    return "x".join(str(x) for x in shape)


def _cells(shape, steps):
    n = 1
    for size in shape:
        n *= size
    return n * steps


def _time_call(fn, *args):
    t0 = time.perf_counter()
    result = fn(*args)
    jax.block_until_ready(result)
    return time.perf_counter() - t0, result


def _bench_case(name, shape, steps, colors, neighborhood, build_step_fn):
    key = jax.random.PRNGKey(sum(shape) + steps + colors)
    state = jax.random.randint(key, shape, 0, colors, dtype=jnp.int32)
    step_fn = build_step_fn(colors)

    def run(initial):
        return ca.evolve(step_fn, initial, steps)

    jitted = jax.jit(run)
    compile_s, _ = _time_call(jitted, state)
    samples = []
    for _ in range(3):
        elapsed, _ = _time_call(jitted, state)
        samples.append(elapsed)

    best = min(samples)
    updates = _cells(shape, steps)
    return {
        "case": name,
        "shape": _format_shape(shape),
        "steps": steps,
        "colors": colors,
        "neighborhood": neighborhood,
        "compile_ms": compile_s * 1000,
        "best_ms": best * 1000,
        "mcells_s": updates / best / 1_000_000,
    }


def _patterned_table(colors, table_size):
    return jnp.mod(jnp.arange(table_size, dtype=jnp.int32), colors)


def _shift_totalistic_1d(radius):
    offsets = ca.offsets_1d(radius)
    weights = ca.totalistic_weights(offsets)

    def build(colors):
        table = _patterned_table(colors, ca.table_size_totalistic(colors, offsets))
        return lambda x: ca.step_shift_accumulate(x, table, offsets, weights)

    return build


def _shift_general_1d(radius):
    offsets = ca.offsets_1d(radius)

    def build(colors):
        weights = ca.general_weights(colors, offsets)
        table = _patterned_table(colors, ca.table_size_general(colors, offsets))
        return lambda x: ca.step_shift_accumulate(x, table, offsets, weights)

    return build


def _conv_totalistic(offsets):
    weights = [1] * len(offsets)
    kernel = ca.kernel_from_offsets(offsets, weights)

    def build(colors):
        table = _patterned_table(colors, ca.table_size_totalistic(colors, offsets))
        return lambda x: ca.step_wrapped_convolution(x, table, kernel)

    return build


def _conv_weighted(offsets):
    weights = [2 if any(coord != 0 for coord in offset) else 1 for offset in offsets]
    kernel = ca.kernel_from_offsets(offsets, weights)

    def build(colors):
        table = _patterned_table(colors, ca.table_size_weighted(colors, weights))
        return lambda x: ca.step_wrapped_convolution(x, table, kernel)

    return build


def _conv_general(offsets):
    def build(colors):
        weights = ca.general_weights(colors, offsets)
        kernel = ca.kernel_from_offsets(offsets, weights)
        table = _patterned_table(colors, ca.table_size_general(colors, offsets))
        return lambda x: ca.step_wrapped_convolution(x, table, kernel)

    return build


def _second_order_1d(radius):
    offsets = ca.offsets_1d(radius)
    weights = ca.totalistic_weights(offsets)

    def build(colors):
        table = _patterned_table(colors, ca.table_size_totalistic(colors, offsets))
        local_step = lambda x: ca.step_shift_accumulate(x, table, offsets, weights)

        def step(pair):
            previous, current = pair
            nxt = ca.reversible_second_order_step(previous, current, local_step, colors)
            return jnp.stack([current, nxt])

        return step

    return build


def _bench_second_order(name, shape, steps, colors, radius):
    key = jax.random.PRNGKey(sum(shape) + steps + colors + 99)
    previous = jax.random.randint(key, shape, 0, colors, dtype=jnp.int32)
    current = jax.random.randint(key + 1, shape, 0, colors, dtype=jnp.int32)
    pair = jnp.stack([previous, current])
    step_fn = _second_order_1d(radius)(colors)

    def run(initial_pair):
        def body(carry, _):
            nxt = step_fn(carry)
            return nxt, nxt[1]

        _, history = jax.lax.scan(body, initial_pair, None, steps)
        return history

    jitted = jax.jit(run)
    compile_s, _ = _time_call(jitted, pair)
    samples = []
    for _ in range(3):
        elapsed, _ = _time_call(jitted, pair)
        samples.append(elapsed)

    best = min(samples)
    return {
        "case": name,
        "shape": _format_shape(shape),
        "steps": steps,
        "colors": colors,
        "neighborhood": f"radius={radius}",
        "compile_ms": compile_s * 1000,
        "best_ms": best * 1000,
        "mcells_s": _cells(shape, steps) / best / 1_000_000,
    }


def _bench_block_2d(shape, steps):
    rotate_clockwise = []
    for code in range(16):
        a, b, c, d = ca.integer_digits(code, 2, 4).tolist()
        rotate_clockwise.append(int(c * 8 + a * 4 + d * 2 + b))
    key = jax.random.PRNGKey(sum(shape) + steps + 123)
    state = jax.random.randint(key, shape, 0, 2, dtype=jnp.int32)

    def run(initial, step_count):
        return ca.block_run(
            rotate_clockwise,
            initial,
            2,
            (2, 2),
            ((0, 0), (1, 1)),
            step_count,
        )

    jitted = jax.jit(run)
    step_count = jnp.asarray(steps, dtype=jnp.int32)
    compile_s, _ = _time_call(jitted, state, step_count)
    samples = []
    for _ in range(3):
        elapsed, _ = _time_call(jitted, state, step_count)
        samples.append(elapsed)

    best = min(samples)
    return {
        "case": "block_margolus_rotate",
        "shape": _format_shape(shape),
        "steps": steps,
        "colors": 2,
        "neighborhood": "block=2x2 phases=2",
        "compile_ms": compile_s * 1000,
        "best_ms": best * 1000,
        "mcells_s": _cells(shape, steps) / best / 1_000_000,
    }


def _bench_many_rules_case(name, state_shape, steps, colors, neighborhood, rule_count, build_step_fn):
    key = jax.random.PRNGKey(sum(state_shape) + steps + colors + rule_count)
    state = jax.random.randint(key, (rule_count,) + state_shape, 0, colors, dtype=jnp.int32)
    step_fn = build_step_fn(colors, rule_count)

    def run(initial):
        def body(carry, _):
            nxt = step_fn(carry)
            return nxt, None

        final, _ = jax.lax.scan(body, initial, None, steps)
        return final

    jitted = jax.jit(run)
    compile_s, _ = _time_call(jitted, state)
    samples = []
    for _ in range(3):
        elapsed, _ = _time_call(jitted, state)
        samples.append(elapsed)

    best = min(samples)
    updates = rule_count * _cells(state_shape, steps)
    return {
        "case": name,
        "rules": rule_count,
        "shape": _format_shape(state_shape),
        "steps": steps,
        "colors": colors,
        "neighborhood": neighborhood,
        "compile_ms": compile_s * 1000,
        "best_ms": best * 1000,
        "mcells_s": updates / best / 1_000_000,
    }


def _many_rules_shift_general_1d(radius):
    offsets = ca.offsets_1d(radius)
    weights = ca.general_weights(2, offsets)
    table_size = ca.table_size_general(2, offsets)

    def build(colors, rule_count):
        if colors != 2:
            raise ValueError("many-rule general 1D benchmark currently uses binary rules")
        rules = jnp.arange(rule_count, dtype=jnp.int32)
        tables = ca.integer_digits_array(rules, colors, table_size)
        return lambda x: ca.step_shift_accumulate_vmap(x, tables, offsets, weights)

    return build


def _many_rules_conv_totalistic(offsets):
    weights = [1] * len(offsets)
    kernel = ca.kernel_from_offsets(offsets, weights)
    table_size = ca.table_size_totalistic(2, offsets)

    def build(colors, rule_count):
        if colors != 2:
            raise ValueError("many-rule totalistic benchmark currently uses binary rules")
        rules = jnp.arange(rule_count, dtype=jnp.int32)
        tables = ca.integer_digits_array(rules, colors, table_size)
        return lambda x: ca.step_wrapped_convolution_vmap(x, tables, kernel)

    return build


def benchmark_ensemble_suite():
    rows = []

    for rule_count in [256, 1024, 4096, 16384]:
        rows.append(
            _bench_many_rules_case(
                "1d_many_rules_general",
                (1024,),
                256,
                2,
                "radius=1",
                rule_count,
                _many_rules_shift_general_1d(1),
            )
        )

    moore = ca.moore_offsets(2, 1)
    for rule_count in [256, 1024, 4096]:
        rows.append(
            _bench_many_rules_case(
                "2d_many_rules_totalistic_moore",
                (64, 64),
                128,
                2,
                "moore_r1_9",
                rule_count,
                _many_rules_conv_totalistic(moore),
            )
        )

    return rows


def benchmark_suite(mode):
    if mode == "ensemble":
        return benchmark_ensemble_suite()

    rows = []

    one_d_widths = [128, 512, 1024] if mode == "full" else [128, 1024]
    two_d_sizes = [32, 64, 128] if mode == "full" else [32, 128]
    three_d_sizes = [12, 24, 32] if mode == "full" else [12, 32]
    steps = 128 if mode == "full" else 64

    for colors in [2, 4]:
        for width in one_d_widths:
            for radius in [1, 2, 4]:
                rows.append(
                    _bench_case(
                        "1d_shift_totalistic",
                        (width,),
                        steps,
                        colors,
                        f"radius={radius}",
                        _shift_totalistic_1d(radius),
                    )
                )
            for radius in [1, 2]:
                rows.append(
                    _bench_case(
                        "1d_shift_general",
                        (width,),
                        steps,
                        colors,
                        f"radius={radius}",
                        _shift_general_1d(radius),
                    )
                )

    for width in one_d_widths:
        rows.append(_bench_second_order("1d_second_order_totalistic", (width,), steps, 2, 1))

    for colors in [2, 4]:
        for size in two_d_sizes:
            moore = ca.moore_offsets(2, 1)
            von_neumann = ca.von_neumann_offsets(2, 1)
            rows.append(
                _bench_case(
                    "2d_conv_totalistic_moore",
                    (size, size),
                    steps,
                    colors,
                    "moore_r1_9",
                    _conv_totalistic(moore),
                )
            )
            rows.append(
                _bench_case(
                    "2d_conv_weighted_moore",
                    (size, size),
                    steps,
                    colors,
                    "moore_r1_9",
                    _conv_weighted(moore),
                )
            )
            rows.append(
                _bench_case(
                    "2d_conv_general_vn",
                    (size, size),
                    steps,
                    colors,
                    "von_neumann_r1_5",
                    _conv_general(von_neumann),
                )
            )

    for size in two_d_sizes:
        rows.append(_bench_block_2d((size, size), min(steps, 64)))

    for colors in [2, 4]:
        for size in three_d_sizes:
            von_neumann = ca.von_neumann_offsets(3, 1)
            moore = ca.moore_offsets(3, 1)
            rows.append(
                _bench_case(
                    "3d_conv_totalistic_vn",
                    (size, size, size),
                    min(steps, 64),
                    colors,
                    "von_neumann_r1_7",
                    _conv_totalistic(von_neumann),
                )
            )
            rows.append(
                _bench_case(
                    "3d_conv_weighted_vn",
                    (size, size, size),
                    min(steps, 64),
                    colors,
                    "von_neumann_r1_7",
                    _conv_weighted(von_neumann),
                )
            )
            rows.append(
                _bench_case(
                    "3d_conv_totalistic_moore",
                    (size, size, size),
                    min(steps, 32),
                    colors,
                    "moore_r1_27",
                    _conv_totalistic(moore),
                )
            )

    return rows


def print_markdown(rows):
    headers = ["case", "rules", "shape", "steps", "colors", "neighborhood", "compile_ms", "best_ms", "mcells_s"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                value = f"{value:.2f}"
            values.append(str(value))
        print("| " + " | ".join(values) + " |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full", "ensemble"], default="quick")
    args = parser.parse_args()
    print(f"JAX {jax.__version__} devices={jax.devices()}")
    print_markdown(benchmark_suite(args.mode))


if __name__ == "__main__":
    main()
