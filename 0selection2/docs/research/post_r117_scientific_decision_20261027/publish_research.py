"""Publish one completed scientific handoff with its Sol feasibility sidecar."""

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "post_r116_major_direction_20261027/publish_research.py"
FILES = ("REQUEST.md", "ROUTING.json", "REPORT.md", "sources.csv", "directions.csv",
         "REPORT.pdf", "render_notes.json", "DELIVERY.md", "render_pdf.py", "publish_research.py",
         ROOT.parents[1] / "experiment_planning/early_graph_context_20261027.md")
MESSAGE = "Publish Astra scientific decision after complete R117 and Sol context feasibility"


def publish():
    spec = importlib.util.spec_from_file_location("post_r117_scientific_publisher", SOURCE)
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    publisher.ROOT, publisher.FILES = ROOT, FILES
    original_git = publisher.git

    def git(*args):
        if args and args[0] == "commit":
            values = list(args)
            values[values.index("-m") + 1] = MESSAGE
            args = tuple(values)
        return original_git(*args)

    publisher.git = git
    return publisher.publish()


if __name__ == "__main__":
    print(json.dumps(publish(), ensure_ascii=False))
