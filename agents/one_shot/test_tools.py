import json
import unittest

from copilot import ToolInvocation

from agents.one_shot import tools


class MathToolTests(unittest.IsolatedAsyncioTestCase):
    async def invoke(self, tool, arguments):
        return await tool.handler(
            ToolInvocation(
                session_id="math-test",
                tool_call_id="test-call",
                tool_name=tool.name,
                arguments=arguments,
            )
        )

    async def success(self, tool, arguments):
        result = await self.invoke(tool, arguments)
        self.assertEqual(result.result_type, "success", result.error)
        return json.loads(result.text_result_for_llm)

    async def test_lagrange_solution_matches_polynomial_checks(self):
        system = await self.success(
            tools.solve_linear_system,
            {
                "matrix": [[6, 2, -1], [2, 4, -1], [1, 1, 0]],
                "vector": [8, 6, 10],
            },
        )
        x, y, multiplier = system["solution"]
        self.assertAlmostEqual(x, 11 / 3)
        self.assertAlmostEqual(y, 19 / 3)
        self.assertAlmostEqual(multiplier, 80 / 3)
        self.assertLess(system["max_abs_residual"], 1e-12)
        self.assertGreaterEqual(system["condition_number"], 1)

        roots = await self.success(
            tools.find_polynomial_roots, {"coefficients": [6, -22]}
        )
        self.assertAlmostEqual(roots["roots"][0]["real"], x)
        self.assertEqual(roots["roots"][0]["imaginary"], 0)
        self.assertLess(roots["max_abs_residual"], 1e-12)

        values = await self.success(
            tools.evaluate_polynomial,
            {"coefficients": [3, -22, 140], "x_values": [x, 0, 10]},
        )
        minimum, left, right = [
            item["value"] for item in values["evaluations"]
        ]
        self.assertAlmostEqual(minimum, 299 / 3)
        self.assertEqual(left, 140)
        self.assertEqual(right, 220)
        self.assertLess(minimum, min(left, right))

    async def test_real_and_complex_polynomial_roots(self):
        real = await self.success(
            tools.find_polynomial_roots, {"coefficients": [1, -3, 2]}
        )
        self.assertEqual(
            sorted(round(root["real"], 10) for root in real["roots"]), [1, 2]
        )
        complex_result = await self.success(
            tools.find_polynomial_roots, {"coefficients": [1, 0, 1]}
        )
        for root in complex_result["roots"]:
            self.assertAlmostEqual(root["real"], 0)
            self.assertAlmostEqual(abs(root["imaginary"]), 1)
        self.assertLess(complex_result["max_abs_residual"], 1e-12)

    async def test_constant_polynomial_evaluation(self):
        result = await self.success(
            tools.evaluate_polynomial,
            {"coefficients": [7], "x_values": [-2, 0, 3]},
        )
        self.assertEqual(
            [entry["value"] for entry in result["evaluations"]], [7, 7, 7]
        )

    async def test_invalid_inputs_and_singular_systems_fail(self):
        cases = [
            (tools.solve_linear_system, {"matrix": [[1, 2]], "vector": [1]}),
            (
                tools.solve_linear_system,
                {"matrix": [[1, 2], [2, 4]], "vector": [1, 2]},
            ),
            (tools.solve_linear_system, {"matrix": [[1]], "vector": []}),
            (tools.find_polynomial_roots, {"coefficients": [0, 1]}),
            (tools.find_polynomial_roots, {"coefficients": [0]}),
            (tools.evaluate_polynomial, {"coefficients": [], "x_values": [1]}),
            (
                tools.evaluate_polynomial,
                {"coefficients": [1, float("inf")], "x_values": [1]},
            ),
            (
                tools.evaluate_polynomial,
                {"coefficients": [1e308, 1], "x_values": [1e308]},
            ),
            (
                tools.evaluate_polynomial,
                {"coefficients": [1], "x_values": [True]},
            ),
        ]
        for tool, arguments in cases:
            with self.subTest(tool=tool.name, arguments=arguments):
                result = await self.invoke(tool, arguments)
                self.assertEqual(result.result_type, "failure")
                self.assertTrue(result.error)


if __name__ == "__main__":
    unittest.main()
