"""
Engine components reach each other through ``self.engine``, never by name strings.

``getattr(self, "name", default)`` on a name that lives on the engine or on
another component does not fail: it silently returns the default. This test
fails on any such access in core/engine/.
"""

import ast
import unittest
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1] / "core" / "engine"


def _own_names(cls):
    names = {n.name for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    names |= {t.id for n in cls.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    names |= {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
    for n in ast.walk(cls):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self" and isinstance(n.ctx, ast.Store):
            names.add(n.attr)
    return names


class TestComponentBoundaries(unittest.TestCase):
    def test_components_do_not_read_foreign_names_by_string(self):
        offenders = []
        for path in sorted(ENGINE.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
                own = _own_names(cls)
                for n in ast.walk(cls):
                    if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)):
                        continue
                    if n.func.id not in ("getattr", "hasattr", "setattr") or len(n.args) < 2:
                        continue
                    obj, name = n.args[0], n.args[1]
                    if (
                        isinstance(obj, ast.Name)
                        and obj.id == "self"
                        and isinstance(name, ast.Constant)
                        and isinstance(name.value, str)
                        and name.value not in own
                    ):
                        offenders.append(f"{path.name}:{n.lineno} {cls.name} {n.func.id}(self, {name.value!r})")
        self.assertEqual(offenders, [])

    def test_components_hand_out_the_engine_not_themselves(self):
        # Tasks and graphs keep an engine reference (task.engine_ref); a
        # component storing itself there gives them an object without the
        # engine's runtime, and every lookup through it falls back silently.
        offenders = []
        for path in sorted(ENGINE.rglob("*.py")):
            for cls in (n for n in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(n, ast.ClassDef)):
                for n in ast.walk(cls):
                    if (
                        isinstance(n, ast.Assign)
                        and isinstance(n.value, ast.Name)
                        and n.value.id == "self"
                        and any(isinstance(t, (ast.Attribute, ast.Subscript)) for t in n.targets)
                    ):
                        offenders.append(f"{path.name}:{n.lineno} {cls.name}: {ast.unparse(n)}")
                    # lambda ..., eng=self: callbacks capture the component the same way
                    if isinstance(n, ast.arguments):
                        for d in list(n.defaults) + [d for d in n.kw_defaults if d is not None]:
                            if isinstance(d, ast.Name) and d.id == "self":
                                offenders.append(f"{path.name}:{d.lineno} {cls.name}: default argument captures self")
        self.assertEqual(offenders, [])

    def test_component_modules_do_not_import_each_other(self):
        # Components are built by the engine and reach each other through it;
        # module-level imports between them would form import cycles. The
        # handlers module is not a component and loads after all of them.
        offenders = []
        not_components = {"__init__.py", "common.py", "handlers.py", "engine.py"}
        for path in sorted(ENGINE.glob("*.py")):
            if path.name in not_components:
                continue
            for node in ast.parse(path.read_text(encoding="utf-8")).body:
                if not isinstance(node, ast.ImportFrom) or node.level != 1:
                    continue
                sibling = (node.module or "").split(".")[0]
                if sibling and sibling not in ("common", path.stem) and (path.parent / f"{sibling}.py").exists():
                    offenders.append(f"{path.name}: from .{node.module} import ...")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
