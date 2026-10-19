from __future__ import annotations

import ast
import unittest
from pathlib import Path


class FeatureBoundaryTests(unittest.TestCase):
    def test_shared_infrastructure_does_not_import_application_or_features(self) -> None:
        root = Path(__file__).resolve().parents[1] / "app" / "shared"
        violations = []
        for path in root.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                           else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                if any(module.startswith(("app.features", "app.application", "app.routers")) for module in modules):
                    violations.append(str(path.relative_to(root)))
        self.assertEqual([], violations)

    def test_features_do_not_import_each_other(self) -> None:
        app_root = Path(__file__).resolve().parents[1] / "app"
        features = {
            "human_pose": app_root / "features" / "human_pose",
            "face_landmark_vf50": app_root / "features" / "face_landmark_vf50",
        }
        violations: list[str] = []
        for feature, root in features.items():
            forbidden = "face_landmark_vf50" if feature == "human_pose" else "human_pose"
            for path in root.rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        modules = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        modules = [node.module or ""]
                    else:
                        continue
                    if any(forbidden in module for module in modules):
                        violations.append(f"{path.relative_to(app_root)} imports {forbidden}")
        self.assertEqual([], violations)


if __name__ == "__main__":
    unittest.main()
