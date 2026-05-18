import unittest

import jax.numpy as jnp

import ca_jax as ca
import ca_d1_totalistic
import ca_d2_totalistic
import ca_eca


class CellularAutomataJaxTest(unittest.TestCase):
    def assert_array_equal(self, actual, expected):
        self.assertEqual(jnp.asarray(actual).tolist(), expected)

    def test_integer_digits_are_fixed_width(self):
        self.assert_array_equal(ca.integer_digits(30, 2, 8), [0, 0, 0, 1, 1, 1, 1, 0])

    def test_rule_30_single_seed(self):
        state = jnp.array([0, 0, 0, 1, 0, 0, 0], dtype=jnp.int32)
        self.assert_array_equal(ca.elementary_step(30, state), [0, 0, 1, 1, 1, 0, 0])

    def test_rule_90_single_seed(self):
        state = jnp.array([0, 0, 0, 1, 0, 0, 0], dtype=jnp.int32)
        self.assert_array_equal(ca.elementary_step(90, state), [0, 0, 1, 0, 1, 0, 0])

    def test_existing_eca_runner_uses_fixed_iteration_helper(self):
        state = jnp.array([0, 0, 0, 1, 0, 0, 0], dtype=jnp.int32)
        history = ca_eca.run_eca2(ca_eca.eca_bitcode(30), state, 1)
        self.assert_array_equal(history, [[0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0]])

    def test_totalistic_1d_shift_accumulate(self):
        offsets = ca.offsets_1d(1)
        weights = ca.totalistic_weights(offsets)
        table = ca.table_from_outputs([0, 1, 2, 0, 1, 2, 0])
        state = jnp.array([0, 1, 2, 0], dtype=jnp.int32)

        self.assert_array_equal(
            ca.step_shift_accumulate(state, table, offsets, weights),
            [1, 0, 0, 2],
        )

    def test_existing_totalistic_1d_module_delegates_to_jax_core(self):
        bitcode = ca_d1_totalistic.totalistic_d1_bitcode(3, 1)(4)
        state = jnp.array([0, 1, 2, 0], dtype=jnp.int32)

        direct = ca_d1_totalistic.totalistic_d1_r1_step_fn(bitcode, state)
        via_core = ca.step_shift_accumulate(
            state,
            bitcode,
            ca.offsets_1d(1),
            ca.totalistic_weights(ca.offsets_1d(1)),
        )
        self.assert_array_equal(direct, via_core.tolist())

    def test_outer_totalistic_1d_shift_accumulate(self):
        offsets = ca.offsets_1d(1)
        weights = ca.outer_totalistic_weights(2, offsets)
        table = ca.table_from_outputs([0, 1, 0, 1, 0, 1])
        state = jnp.array([0, 1, 1, 0], dtype=jnp.int32)

        self.assert_array_equal(
            ca.step_shift_accumulate(state, table, offsets, weights),
            [0, 1, 1, 0],
        )

    def test_shift_accumulate_vmap_matches_individual_rules(self):
        offsets = ca.offsets_1d(1)
        weights = ca.general_weights(2, offsets)
        tables = jnp.stack(
            [
                ca.wolfram_rule_table(30, 2, 8),
                ca.wolfram_rule_table(90, 2, 8),
            ]
        )
        states = jnp.stack(
            [
                jnp.array([0, 0, 0, 1, 0, 0, 0], dtype=jnp.int32),
                jnp.array([0, 0, 0, 1, 0, 0, 0], dtype=jnp.int32),
            ]
        )

        batched = ca.step_shift_accumulate_vmap(states, tables, offsets, weights)
        expected = jnp.stack(
            [
                ca.step_shift_accumulate(states[0], tables[0], offsets, weights),
                ca.step_shift_accumulate(states[1], tables[1], offsets, weights),
            ]
        )
        self.assert_array_equal(batched, expected.tolist())

    def test_same_state_shift_accumulate_vmap_matches_individual_rules(self):
        offsets = ca.offsets_1d(1)
        weights = ca.general_weights(2, offsets)
        tables = jnp.stack(
            [
                ca.wolfram_rule_table(30, 2, 8),
                ca.wolfram_rule_table(90, 2, 8),
            ]
        )
        state = jnp.array([0, 0, 0, 1, 0, 0, 0], dtype=jnp.int32)

        batched = ca.step_shift_accumulate_same_state_vmap(state, tables, offsets, weights)
        expected = jnp.stack(
            [
                ca.step_shift_accumulate(state, tables[0], offsets, weights),
                ca.step_shift_accumulate(state, tables[1], offsets, weights),
            ]
        )
        self.assert_array_equal(batched, expected.tolist())

    def test_2d_convolution_matches_shift_accumulate_for_moore_totalistic(self):
        offsets = ca.moore_offsets(2, 1)
        weights = ca.totalistic_weights(offsets)
        kernel = ca.kernel_from_offsets(offsets, weights)
        state = jnp.array(
            [
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
            ],
            dtype=jnp.int32,
        )

        shift_codes = ca.encode_shift_accumulate(state, offsets, weights)
        conv_codes = ca.encode_wrapped_convolution(state, kernel)
        self.assert_array_equal(conv_codes, shift_codes.tolist())

    def test_3d_convolution_matches_shift_accumulate_for_von_neumann_totalistic(self):
        offsets = ca.von_neumann_offsets(3, 1)
        weights = ca.totalistic_weights(offsets)
        kernel = ca.kernel_from_offsets(offsets, weights)
        state = jnp.arange(27, dtype=jnp.int32).reshape((3, 3, 3)) % 2

        shift_codes = ca.encode_shift_accumulate(state, offsets, weights)
        conv_codes = ca.encode_wrapped_convolution(state, kernel)
        self.assert_array_equal(conv_codes, shift_codes.tolist())

    def test_existing_2d_totalistic_module_uses_wrapped_convolution(self):
        bitcode = ca_d2_totalistic.totalistic_d2n9_bitcode_fn(2)(0)
        state = jnp.array(
            [
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
            ],
            dtype=jnp.int32,
        )

        self.assert_array_equal(
            ca_d2_totalistic.totalistic_d2n9_step_fn(bitcode, state),
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        )

    def test_wrapped_convolution_vmap_matches_individual_rules(self):
        offsets = ca.moore_offsets(2, 1)
        weights = ca.totalistic_weights(offsets)
        kernel = ca.kernel_from_offsets(offsets, weights)
        tables = jnp.stack(
            [
                ca.table_from_outputs([0, 1] * 5),
                ca.table_from_outputs([1, 0] * 5),
            ]
        )
        states = jnp.stack(
            [
                jnp.array(
                    [
                        [0, 1, 0],
                        [1, 1, 0],
                        [0, 0, 1],
                    ],
                    dtype=jnp.int32,
                ),
                jnp.array(
                    [
                        [1, 0, 1],
                        [0, 1, 1],
                        [1, 0, 0],
                    ],
                    dtype=jnp.int32,
                ),
            ]
        )

        batched = ca.step_wrapped_convolution_vmap(states, tables, kernel)
        expected = jnp.stack(
            [
                ca.step_wrapped_convolution(states[0], tables[0], kernel),
                ca.step_wrapped_convolution(states[1], tables[1], kernel),
            ]
        )
        self.assert_array_equal(batched, expected.tolist())

    def test_game_of_life_blinker_with_outer_totalistic_table(self):
        offsets = ca.moore_offsets(2, 1)
        weights = ca.outer_totalistic_weights(2, offsets)
        kernel = ca.kernel_from_offsets(offsets, weights)

        outputs = [0] * ca.table_size_outer_totalistic(2, offsets)
        outputs[2 * 3 + 0] = 1
        outputs[2 * 2 + 1] = 1
        outputs[2 * 3 + 1] = 1
        table = ca.table_from_outputs(outputs)

        state = jnp.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=jnp.int32,
        )

        expected = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        self.assert_array_equal(ca.step_wrapped_convolution(state, table, kernel), expected)
        self.assert_array_equal(ca_d2_totalistic.game_of_life_step_fn(state), expected)

    def test_second_order_reversible_rule_round_trips(self):
        previous = jnp.array([0, 1, 0, 0, 1, 1, 0], dtype=jnp.int32)
        current = jnp.array([1, 0, 0, 1, 0, 1, 1], dtype=jnp.int32)
        local_step = lambda x: ca.elementary_step(30, x)

        history = ca.reversible_second_order_evolve(local_step, previous, current, 5, 2)

        reconstructed = [history[-1], history[-2]]
        next_state = history[-1]
        cur = history[-2]
        for _ in range(5):
            prev = ca.reversible_second_order_reverse_step(cur, next_state, local_step, 2)
            reconstructed.append(prev)
            next_state, cur = cur, prev

        self.assert_array_equal(reconstructed[::-1], history.tolist())

    def test_second_order_reversible_step_is_mod_k(self):
        previous = jnp.array([0, 1, 2, 1, 0], dtype=jnp.int32)
        current = jnp.array([1, 2, 0, 2, 1], dtype=jnp.int32)
        offsets = ca.offsets_1d(1)
        weights = ca.totalistic_weights(offsets)
        table = ca.table_from_outputs([0, 1, 2, 0, 1, 2, 0])
        local_step = lambda x: ca.step_shift_accumulate(x, table, offsets, weights)

        next_state = ca.reversible_second_order_step(previous, current, local_step, 3)
        recovered = ca.reversible_second_order_reverse_step(current, next_state, local_step, 3)

        self.assert_array_equal(recovered, previous.tolist())

    def test_1d_block_rule_phase_and_inverse(self):
        swap_bits = (0, 2, 1, 3)
        state = jnp.array([0, 1, 1, 0], dtype=jnp.int32)

        phase0 = ca.apply_block_table(state, swap_bits, 2, (2,), (0,))
        phase1 = ca.apply_block_table(state, swap_bits, 2, (2,), (1,))

        self.assert_array_equal(phase0, [1, 0, 0, 1])
        self.assert_array_equal(phase1, [0, 1, 1, 0])
        self.assert_array_equal(
            ca.apply_block_table(phase0, ca.inverse_permutation(swap_bits), 2, (2,), (0,)),
            state.tolist(),
        )

    def test_2d_margolus_block_rule_round_trips(self):
        rotate_clockwise = []
        for code in range(16):
            a, b, c, d = ca.integer_digits(code, 2, 4).tolist()
            rotate_clockwise.append(int(c * 8 + a * 4 + d * 2 + b))
        rotate_counterclockwise = ca.inverse_permutation(rotate_clockwise)

        state = jnp.array(
            [
                [0, 1, 1, 0],
                [1, 0, 0, 1],
                [0, 0, 1, 1],
                [1, 1, 0, 0],
            ],
            dtype=jnp.int32,
        )

        phases = ((0, 0), (1, 1))
        forward = ca.block_evolve(rotate_clockwise, state, 2, (2, 2), phases, 4)
        current = forward[-1]
        for phase in reversed([phases[i % len(phases)] for i in range(4)]):
            current = ca.apply_block_table(current, rotate_counterclockwise, 2, (2, 2), phase)

        self.assert_array_equal(current, state.tolist())

    def test_block_run_matches_history_evolve_final_state(self):
        swap_bits = (0, 2, 1, 3)
        state = jnp.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=jnp.int32)
        phases = ((0,), (1,))

        history = ca.block_evolve(swap_bits, state, 2, (2,), phases, 5)
        final = ca.block_run(swap_bits, state, 2, (2,), phases, jnp.asarray(5, dtype=jnp.int32))

        self.assert_array_equal(final, history[-1].tolist())


if __name__ == "__main__":
    unittest.main()
