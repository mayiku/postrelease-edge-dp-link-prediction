from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "scripts" / "train_postrelease.py"
FORBIDDEN_MODULES = {
    "src.raw_graph",
    "src.postrelease_lp",
    "scipy.io",
}
FORBIDDEN_NAMES = {
    "load_graph_data",
    "graph_edges",
    "loadmat",
}


def module_file(module_name: str) -> Path | None:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin)


def imported_local_modules(tree: ast.AST) -> set[str]:
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(("src.", "scripts.")):
                    result.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith(("src.", "scripts.")):
                result.add(node.module)
    return result


def audit_entrypoint() -> tuple[list[str], set[str]]:
    pending = [ENTRYPOINT]
    visited: set[Path] = set()
    modules: set[str] = set()
    violations: list[str] = []
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modules.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    modules.add(node.module)
                for alias in node.names:
                    if alias.name in FORBIDDEN_NAMES:
                        violations.append(
                            f"{path}: imports {alias.name}"
                        )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in FORBIDDEN_NAMES:
                    violations.append(
                        f"{path}: defines {node.name}"
                    )
            elif isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in FORBIDDEN_NAMES
                ):
                    violations.append(
                        f"{path}: calls {node.func.attr}"
                    )
        for module_name in imported_local_modules(tree):
            if module_name in FORBIDDEN_MODULES:
                violations.append(
                    f"{path}: imports forbidden module {module_name}"
                )
                continue
            if not module_name.startswith("src."):
                continue
            resolved = module_file(module_name)
            if resolved is not None:
                pending.append(resolved)
    return sorted(set(violations)), modules


class ImportBoundaryTests(unittest.TestCase):
    def test_training_import_chain_has_no_graph_io(self) -> None:
        violations, modules = audit_entrypoint()
        self.assertFalse(
            violations,
            "training import graph crosses the release boundary:\n"
            + "\n".join(violations),
        )
        self.assertNotIn("src.postrelease_lp", modules)
        self.assertNotIn("src.raw_graph", modules)


if __name__ == "__main__":
    unittest.main()
