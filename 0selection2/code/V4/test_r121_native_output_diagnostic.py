"""Only verify instrumentation/trace preservation, not real GPU recovery."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import r121_native_output_diagnostic as diagnostic


class DiagnosticTests(unittest.TestCase):
    def test_literal_root_derives_output_without_loading_native_runtime(self):
        root = diagnostic.ROOT / 'argparse_check'
        with patch.object(diagnostic, 'run', return_value={'solver_execution': False}) as run:
            self.assertEqual(diagnostic.main(['--root', str(root)]), 0)
        args = run.call_args.args[0]
        self.assertEqual(args.root, root.resolve())
        self.assertEqual(args.output, root.resolve() / 'ATSP__UNICO_MatPOENet_attempt1.json')
        self.assertFalse(args.execute)

    def test_full_actual_exception_chain_is_retained_and_native_called_once(self):
        def failing_native(items):
            raise TypeError('explicit CPU instrumentation fixture, NOT a GPU recovery witness')

        backend = SimpleNamespace(run=failing_native)
        session = SimpleNamespace(backend=backend)
        state = dict(native_calls=0)

        def original_helper(session, manifest, task):
            return session.backend.run(['original item'])

        with patch.object(diagnostic.base, 'solve_native', side_effect=original_helper):
            with self.assertRaises(TypeError):
                diagnostic.native_once(session, {}, {}, state)
        self.assertEqual(state['native_calls'], 1)
        self.assertEqual(state['stage'], 'native_backend.run')
        self.assertIn('failing_native', state['native_backend_exception']['traceback'])
        self.assertTrue(any(f['name'] == 'failing_native' for f in state['native_backend_exception']['frames']))
        self.assertIs(backend.run, failing_native)


if __name__ == '__main__':
    unittest.main()
