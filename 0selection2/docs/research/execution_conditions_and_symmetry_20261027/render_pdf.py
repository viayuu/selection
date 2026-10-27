"""Mechanical unchanged-source PDF for the completed Astra scientific sidecar."""

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'post_r117_scientific_decision_20261027/render_pdf.py'


def main():
    spec = importlib.util.spec_from_file_location('execution_conditions_pdf', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.render()
    notes = json.loads((ROOT / 'render_notes.json').read_text())
    notes['wrapper_sha256'] = module.sha(__file__)
    notes['completed_scientific_cli_exit_code'] = json.loads((ROOT / 'cli_exit.json').read_text())['exit_code']
    (ROOT / 'render_notes.json').write_text(json.dumps(notes, ensure_ascii=False, indent=2) + '\n')
    (ROOT / 'DELIVERY.md').write_text(
        '# Execution Conditions And Symmetry\n\n'
        f"REPORT.pdf: {notes['pdf']['pages']} pages; three primary papers.\n\n"
        'Astra/max authored the scientific files in a persistent CLI invocation; '
        'Sol rendered them without changing scientific source bytes. Earlier capacity '
        'and missing-agent-handle attempts are retained in ROUTING.json.\n\n'
        'Conclusion: NO_READY_MAIN_DIRECTION. The existing TSP anchor and endogenous '
        'CVRP TopK are not newly missing input roles; fresh MATNET roles do not recover '
        'historical label realizations. No accuracy ceiling is estimated.\n\n'
        'R119 operational process preparation is separate and is not an Astra endorsement '
        'of a ready main direction. It retains the full18 decision scope but excludes '
        'unmatched ATSP process targets; no selector improvement has been measured.\n\n'
        'The continuing whole18 improvement goal remains ACTIVE.\n')


if __name__ == '__main__':
    main()
