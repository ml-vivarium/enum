import os
import unittest

import jax.numpy as jnp

import ca_d1_totalistic
import ca_d2_totalistic
import ca_jax as ca
import wolfram_reference


@unittest.skipUnless(
    os.path.exists(wolfram_reference.DEFAULT_KERNEL),
    "WolframKernel is not installed at the expected app-bundle path",
)
class WolframReferenceTest(unittest.TestCase):
    def assert_matches_wolfram(self, actual, expression):
        expected = wolfram_reference.cellular_automaton(expression)
        self.assertEqual(jnp.asarray(actual).tolist(), expected)

    def test_rule_30_matches_wolfram(self):
        state = jnp.array([0, 0, 0, 1, 0, 0, 0], dtype=jnp.int32)
        actual = ca.evolve(lambda x: ca.elementary_step(30, x), state, 3)

        self.assert_matches_wolfram(actual, "[30, {0,0,0,1,0,0,0}, 3]")

    def test_rule_90_matches_wolfram(self):
        state = jnp.array([0, 0, 0, 1, 0, 0, 0], dtype=jnp.int32)
        actual = ca.evolve(lambda x: ca.elementary_step(90, x), state, 3)

        self.assert_matches_wolfram(actual, "[90, {0,0,0,1,0,0,0}, 3]")

    def test_1d_general_k3_radius1_matches_wolfram(self):
        state = jnp.array([0, 1, 2, 0, 1], dtype=jnp.int32)
        offsets = ca.offsets_1d(1)
        weights = ca.general_weights(3, offsets)
        table = ca.wolfram_rule_table(777, 3, ca.table_size_general(3, offsets))
        actual = ca.evolve(lambda x: ca.step_shift_accumulate(x, table, offsets, weights), state, 2)

        self.assert_matches_wolfram(actual, "[{777,3,1}, {0,1,2,0,1}, 2]")

    def test_1d_general_radius2_matches_wolfram(self):
        state = jnp.array([0, 0, 1, 0, 0, 1, 0], dtype=jnp.int32)
        offsets = ca.offsets_1d(2)
        weights = ca.general_weights(2, offsets)
        table = ca.wolfram_rule_table(12345, 2, ca.table_size_general(2, offsets))
        actual = ca.evolve(lambda x: ca.step_shift_accumulate(x, table, offsets, weights), state, 2)

        self.assert_matches_wolfram(actual, "[{12345,2,2}, {0,0,1,0,0,1,0}, 2]")

    def test_1d_totalistic_matches_wolfram(self):
        state = jnp.array([0, 1, 2, 0, 1], dtype=jnp.int32)
        rule = 777
        bitcode = ca_d1_totalistic.totalistic_d1_bitcode(3, 1)(rule)
        actual = ca.evolve(
            lambda x: ca_d1_totalistic.totalistic_d1_r1_step_fn(bitcode, x),
            state,
            2,
        )

        self.assert_matches_wolfram(actual, "[{777,{3,1},1}, {0,1,2,0,1}, 2]")

    def test_1d_weighted_outer_style_matches_wolfram(self):
        state = jnp.array([0, 1, 1, 0, 1], dtype=jnp.int32)
        offsets = ca.offsets_1d(1)
        weights = ca.outer_totalistic_weights(2, offsets)
        table = ca.wolfram_rule_table(42, 2, ca.table_size_weighted(2, weights.tolist()))
        actual = ca.evolve(lambda x: ca.step_shift_accumulate(x, table, offsets, weights), state, 2)

        self.assert_matches_wolfram(actual, "[{42,{2,{2,1,2}},1}, {0,1,1,0,1}, 2]")

    def test_2d_general_explicit_von_neumann_matches_wolfram(self):
        state = jnp.array(
            [
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
            ],
            dtype=jnp.int32,
        )
        offsets = ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0))
        weights = ca.general_weights(2, offsets)
        kernel = ca.kernel_from_offsets(offsets, weights)
        table = ca.wolfram_rule_table(17, 2, ca.table_size_general(2, offsets))
        actual = ca.evolve(lambda x: ca.step_wrapped_convolution(x, table, kernel), state, 2)

        self.assert_matches_wolfram(
            actual,
            "[{17,2,{{-1,0},{0,-1},{0,0},{0,1},{1,0}}}, {{0,1,0},{1,1,0},{0,0,1}}, 2]",
        )

    def test_2d_totalistic_moore_matches_wolfram(self):
        state = jnp.array(
            [
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
            ],
            dtype=jnp.int32,
        )
        offsets = ca.moore_offsets(2, 1)
        weights = ca.totalistic_weights(offsets)
        kernel = ca.kernel_from_offsets(offsets, weights)
        table = ca.wolfram_rule_table(42, 2, ca.table_size_totalistic(2, offsets))
        actual = ca.evolve(lambda x: ca.step_wrapped_convolution(x, table, kernel), state, 2)

        self.assert_matches_wolfram(
            actual,
            "[{42,{2,1},{{-1,-1},{-1,0},{-1,1},{0,-1},{0,0},{0,1},{1,-1},{1,0},{1,1}}}, {{0,1,0},{1,1,0},{0,0,1}}, 2]",
        )

    def test_2d_weighted_moore_matches_wolfram(self):
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
        offsets = ca.moore_offsets(2, 1)
        weights = [2, 2, 2, 2, 1, 2, 2, 2, 2]
        kernel = ca.kernel_from_offsets(offsets, weights)
        table = ca.wolfram_rule_table(6152, 2, ca.table_size_weighted(2, weights))
        actual = ca.evolve(lambda x: ca.step_wrapped_convolution(x, table, kernel), state, 2)

        self.assert_matches_wolfram(
            actual,
            "[{6152,{2,{{2,2,2},{2,1,2},{2,2,2}}},{1,1}}, {{0,0,0,0,0},{0,0,1,0,0},{0,0,1,0,0},{0,0,1,0,0},{0,0,0,0,0}}, 2]",
        )

    def test_3d_general_explicit_von_neumann_matches_wolfram(self):
        state = jnp.array(
            [
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 1, 0], [1, 1, 0], [0, 0, 1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            dtype=jnp.int32,
        )
        offsets = ((0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
        weights = ca.general_weights(2, offsets)
        kernel = ca.kernel_from_offsets(offsets, weights)
        table = ca.wolfram_rule_table(42, 2, ca.table_size_general(2, offsets))
        actual = ca.evolve(lambda x: ca.step_wrapped_convolution(x, table, kernel), state, 1)

        self.assert_matches_wolfram(
            actual,
            "[{42,2,{{0,0,0},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}}}, {{{0,0,0},{0,1,0},{0,0,0}},{{0,1,0},{1,1,0},{0,0,1}},{{0,0,0},{0,0,0},{0,0,0}}}, 1]",
        )

    def test_3d_totalistic_von_neumann_matches_wolfram(self):
        state = jnp.array(
            [
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 1, 0], [1, 1, 0], [0, 0, 1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            dtype=jnp.int32,
        )
        offsets = ((0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
        weights = ca.totalistic_weights(offsets)
        kernel = ca.kernel_from_offsets(offsets, weights)
        table = ca.wolfram_rule_table(42, 2, ca.table_size_totalistic(2, offsets))
        actual = ca.evolve(lambda x: ca.step_wrapped_convolution(x, table, kernel), state, 1)

        self.assert_matches_wolfram(
            actual,
            "[{42,{2,1},{{0,0,0},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}}}, {{{0,0,0},{0,1,0},{0,0,0}},{{0,1,0},{1,1,0},{0,0,1}},{{0,0,0},{0,0,0},{0,0,0}}}, 1]",
        )

    def test_game_of_life_matches_wolfram(self):
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
        actual = ca.evolve(ca_d2_totalistic.game_of_life_step_fn, state, 2)

        self.assert_matches_wolfram(actual, '["GameOfLife", {{0,0,0,0,0},{0,0,1,0,0},{0,0,1,0,0},{0,0,1,0,0},{0,0,0,0,0}}, 2]')


if __name__ == "__main__":
    unittest.main()
