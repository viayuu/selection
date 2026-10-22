"""Exercise the actual report Markdown fixture with the real list-returning helper."""

import ast
from pathlib import Path
import unittest

from . import r97_reporting as previous


class MarkdownRegression(unittest.TestCase):
    def test_actual_report_fixture(self):
        path = Path(__file__).with_name('r100_reporting.py')
        report = next(node for node in ast.parse(path.read_text()).body
                      if isinstance(node, ast.FunctionDef) and node.name == 'report')
        assignments = {node.targets[0].id: node.value for node in ast.walk(report)
                       if isinstance(node, ast.Assign) and len(node.targets) == 1
                       and isinstance(node.targets[0], ast.Name)}
        keys = ast.literal_eval(assignments['keys'])
        row = dict(zip(keys, ['TSP', 'TEST', 4500, 72.5, 10., 11., 9., -9.09, 11.11, 11.12]))
        fixture = previous._md_table(keys, [[row[key] for key in keys]])
        self.assertIsInstance(fixture, list)
        with self.assertRaises(TypeError):
            '\n'.join(['header', fixture])
        lines = eval(compile(ast.Expression(assignments['lines']), str(path), 'eval'),
                     dict(previous=previous, NOTES=(), keys=keys, metrics=[row]))
        markdown = '\n'.join(lines)
        self.assertTrue(all(isinstance(line, str) for line in lines))
        self.assertIn('\n'.join(fixture), markdown)
        self.assertIn('| TSP | TEST |', markdown)


if __name__ == '__main__':
    unittest.main()
