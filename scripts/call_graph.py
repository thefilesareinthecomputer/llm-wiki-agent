#!/usr/bin/env python3
"""
Call Graph Analyzer for this codebase

Uses Python AST to build call graphs and detect dead code.
Properly handles class nesting, self.method() calls, and framework entry points.

IMPORTANT: This script is a HINT, not ground truth. It cannot do type inference,
so attribute calls on variables (e.g. `gateway.chat()`) cannot be resolved to
exact class methods. Always manually validate results before acting on them.

Usage:
    python scripts/call_graph.py [src_dir]

Output:
    - Import dependency graph
    - Call graph with caller/callee
    - Entry points (framework-called functions)
    - Dead code (no callers, not entry points)
"""

import ast
import sys
from pathlib import Path
from collections import defaultdict


class CodeVisitor(ast.NodeVisitor):
    """AST visitor that tracks class nesting context properly."""

    # Decorators that mark entry points (framework calls these)
    ENTRY_POINT_DECORATORS = {
        "app.get", "app.post", "app.put", "app.delete", "app.patch",
        "router.get", "router.post", "router.put", "router.delete", "router.patch",
        "app.websocket",
    }

    # Method names that are framework callbacks (not called directly)
    FRAMEWORK_CALLBACKS = {
        "on_modified", "on_created", "on_deleted", "on_moved",
        "on_any_event",  # watchdog
        "on_startup", "on_shutdown",  # FastAPI legacy
    }

    def __init__(self, rel_path: str):
        self.rel_path = rel_path
        self.definitions = {}       # (qualified_name, kind) -> (file, lineno)
        self.callers = defaultdict(list)  # (name, kind) -> [(file, lineno, call_type)]
        self.entry_points = set()   # (qualified_name, kind) tuples
        self.imports = []           # list of imported module.names
        self._class_stack = []      # tracks nesting: ["ClassName", ...]
        self._function_stack = []   # tracks function nesting for self-call detection

    def _qualified_name(self, name: str) -> str:
        """Build qualified name from class stack."""
        if self._class_stack:
            return f"{'.'.join(self._class_stack)}.{name}"
        return name

    def _is_entry_point_decorator(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Check if a function has a framework entry-point decorator."""
        for dec in node.decorator_list:
            # @app.get("/path") or @app.post("/path")
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                dec_str = ""
                if isinstance(dec.func.value, ast.Name):
                    dec_str = f"{dec.func.value.id}.{dec.func.attr}"
                if dec_str in self.ENTRY_POINT_DECORATORS:
                    return True
            # @app.get without call (unlikely but handle)
            elif isinstance(dec, ast.Attribute):
                dec_str = ""
                if isinstance(dec.value, ast.Name):
                    dec_str = f"{dec.value.id}.{dec.attr}"
                if dec_str in self.ENTRY_POINT_DECORATORS:
                    return True
        return False

    def visit_ClassDef(self, node: ast.ClassDef):
        qname = self._qualified_name(node.name)
        self.definitions[(qname, "class")] = (self.rel_path, node.lineno)
        self._class_stack.append(node.name)
        self.generic_visit(node)  # visit children WITH class in stack
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._visit_function(node)

    def _visit_function(self, node):
        qname = self._qualified_name(node.name)
        kind = "method" if self._class_stack else "function"
        self.definitions[(qname, kind)] = (self.rel_path, node.lineno)

        # Check for entry-point decorators
        if self._is_entry_point_decorator(node):
            self.entry_points.add((qname, kind))
        # main() and top-level async main() are entry points
        elif node.name == "main" and not self._class_stack:
            self.entry_points.add((qname, kind))
        # Framework callback methods (watchdog, FastAPI lifecycle)
        elif node.name in self.FRAMEWORK_CALLBACKS:
            self.entry_points.add((qname, kind))
        # lifespan() passed as FastAPI(lifespan=) param
        elif node.name == "lifespan" and not self._class_stack:
            self.entry_points.add((qname, kind))

        self._function_stack.append(node.name)
        self.generic_visit(node)  # visit function body
        self._function_stack.pop()

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            # Direct call: foo()
            self.callers[(node.func.id, "any")].append(
                (self.rel_path, node.lineno, "direct")
            )
        elif isinstance(node.func, ast.Attribute):
            # Attribute call: obj.method()
            if isinstance(node.func.value, ast.Name):
                var_name = node.func.value.id
                method_name = node.func.attr

                # self.method() inside a class — resolve to ClassName.method
                if var_name == "self" and self._class_stack:
                    class_method = f"{self._class_stack[-1]}.{method_name}"
                    self.callers[(class_method, "self_call")].append(
                        (self.rel_path, node.lineno, "self_attr")
                    )
                else:
                    # Variable call: gateway.chat()
                    # Record both bare attr and qualified var.attr form
                    self.callers[(method_name, "any")].append(
                        (self.rel_path, node.lineno, "attribute")
                    )
                    qcall = f"{var_name}.{method_name}"
                    self.callers[(qcall, "qualified")].append(
                        (self.rel_path, node.lineno, "qualified_attr")
                    )
            else:
                # Chained call: obj.attr.method() — just record bare attr
                self.callers[(node.func.attr, "any")].append(
                    (self.rel_path, node.lineno, "chained_attr")
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ""
        for alias in node.names:
            self.imports.append(f"{module}.{alias.name}")
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)


def analyze_codebase(src_dir: Path):
    """Analyze all Python files in src_dir using proper nesting-aware visitor."""
    files = list(src_dir.rglob("*.py"))
    # Skip __pycache__
    files = [f for f in files if "__pycache__" not in str(f)]

    all_definitions = {}
    all_callers = defaultdict(list)
    all_entry_points = set()
    file_imports = {}  # rel_path -> [imports]

    for f in files:
        try:
            tree = ast.parse(f.read_text())
            rel_path = str(f.relative_to(src_dir))
            visitor = CodeVisitor(rel_path)
            visitor.visit(tree)

            all_definitions.update(visitor.definitions)
            for key, val_list in visitor.callers.items():
                all_callers[key].extend(val_list)
            all_entry_points.update(visitor.entry_points)
            if visitor.imports:
                file_imports[rel_path] = visitor.imports
        except Exception as e:
            print(f"Error parsing {f}: {e}", file=sys.stderr)

    return all_definitions, all_callers, all_entry_points, file_imports


def find_dead_code(definitions, callers, entry_points, exclude_private=True):
    """Find functions/classes with no callers (not entry points).

    Matching strategy for methods:
    - self.method() calls: ("ClassName.method", "self_call") — exact
    - Variable calls: ("var.method", "qualified") — when var name matches class name
    - Bare name ("method", "any"): ONLY when the method name is unique across all
      classes (no ambiguity). Ambiguous names like "search", "run" require
      self_call or qualified match to avoid false positives.

    Matching strategy for functions:
    - Use ("function_name", "any") — function names are usually unique per module

    __init__ methods: reachable if the class is instantiated.
    """
    dead = []

    # Build set of class names that are instantiated
    instantiated_classes = set()
    for (name, kind), caller_list in callers.items():
        if kind == "any":
            for c_file, c_line, c_type in caller_list:
                if c_type == "direct":
                    if (name, "class") in definitions:
                        instantiated_classes.add(name)

    # Build map: bare method name -> list of qualified names (to detect ambiguity)
    method_name_owners = defaultdict(list)  # "search" -> ["ConversationStore.search", "KBIndex.search", ...]
    for (name, kind) in definitions:
        if kind == "method" and "." in name:
            bare = name.split(".")[-1]
            method_name_owners[bare].append(name)

    # A bare method name is "unambiguous" if only ONE class defines it
    unambiguous_methods = {
        bare for bare, owners in method_name_owners.items() if len(owners) == 1
    }

    for (name, kind), (file, lineno) in definitions.items():
        if exclude_private and name.startswith("_"):
            continue
        if (name, kind) in entry_points:
            continue

        has_callers = False

        if kind == "method":
            self_calls = callers.get((name, "self_call"), [])
            qualified_calls = callers.get((name, "qualified"), [])

            bare_name = name.split(".")[-1] if "." in name else name
            bare_any_calls = []

            # Only use bare name matching when unambiguous
            if bare_name in unambiguous_methods:
                bare_any_calls = callers.get((bare_name, "any"), [])

            # Also check if bare name is a standalone function
            if (bare_name, "function") in definitions:
                bare_any_calls += [c for c in callers.get((bare_name, "any"), [])
                                   if c[2] == "direct"]

            all_callers = self_calls + qualified_calls + bare_any_calls
            has_callers = bool(all_callers)

        elif kind == "function":
            all_callers = list(callers.get((name, "any"), []))
            all_callers += list(callers.get((name, "qualified"), []))
            has_callers = bool(all_callers)

        elif kind == "class":
            all_callers = callers.get((name, "any"), [])
            has_callers = bool(all_callers)

        # __init__ is reachable if the class is instantiated
        if name.endswith(".__init__"):
            class_name = name.rsplit(".", 1)[0]
            if class_name in instantiated_classes:
                has_callers = True

        if not has_callers:
            dead.append((name, kind, file, lineno))

    return dead


def _gather_callers_for_display(name, kind, callers, definitions, method_name_owners=None):
    """Gather callers for display using same logic as find_dead_code."""
    all_callers = []

    if method_name_owners is None:
        method_name_owners = {}

    # Build unambiguous set
    unambiguous_methods = {
        bare for bare, owners in method_name_owners.items() if len(owners) == 1
    } if method_name_owners else set()

    if kind == "method":
        all_callers.extend(callers.get((name, "self_call"), []))
        all_callers.extend(callers.get((name, "qualified"), []))
        bare_name = name.split(".")[-1] if "." in name else name
        if bare_name in unambiguous_methods:
            all_callers.extend(callers.get((bare_name, "any"), []))
        if (bare_name, "function") in definitions:
            all_callers.extend([c for c in callers.get((bare_name, "any"), [])
                                if c[2] == "direct"])
    elif kind == "function":
        all_callers.extend(callers.get((name, "any"), []))
        all_callers.extend(callers.get((name, "qualified"), []))
    elif kind == "class":
        all_callers.extend(callers.get((name, "any"), []))

    # Deduplicate by (file, line)
    seen = set()
    unique = []
    for c in all_callers:
        key = (c[0], c[1])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def print_import_graph(file_imports):
    """Print import dependency graph."""
    print("=== IMPORT DEPENDENCIES ===\n")

    for rel_path in sorted(file_imports.keys()):
        imports = file_imports[rel_path]
        print(f"{rel_path}:")
        for imp in imports[:10]:
            print(f"  -> {imp}")
        if len(imports) > 10:
            print(f"  ... +{len(imports) - 10} more")
        print()


def print_call_graph(definitions, callers, entry_points):
    """Print the call graph with entry point awareness."""
    print("=== CALL GRAPH ===\n")

    # Build method_name_owners for ambiguity detection
    method_name_owners = defaultdict(list)
    for (name, kind) in definitions:
        if kind == "method" and "." in name:
            bare = name.split(".")[-1]
            method_name_owners[bare].append(name)

    ep_names = {name for name, kind in entry_points}

    for (name, kind), (file, lineno) in sorted(definitions.items()):
        is_entry = name in ep_names
        unique_callers = _gather_callers_for_display(
            name, kind, callers, definitions, method_name_owners
        )

        prefix = "[ENTRY] " if is_entry else ""
        print(f"{prefix}{kind}: {name} @ {file}:{lineno}")

        if is_entry:
            print(f"  -> Called by: framework/decorator")
        elif unique_callers:
            caller_str = ", ".join([f"{c[0]}:{c[1]}" for c in unique_callers[:4]])
            if len(unique_callers) > 4:
                caller_str += f" (+{len(unique_callers) - 4})"
            print(f"  -> Called by: {caller_str}")
        else:
            print(f"  -> ** NO CALLERS **")
        print()


def main():
    src_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("src")

    if not src_dir.exists():
        print(f"Error: {src_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    definitions, callers, entry_points, file_imports = analyze_codebase(src_dir)

    # 1. Import graph
    print_import_graph(file_imports)

    # 2. Call graph
    print_call_graph(definitions, callers, entry_points)

    # 3. Entry points summary
    print("=== ENTRY POINTS (framework-called) ===\n")
    for name, kind in sorted(entry_points):
        info = definitions.get((name, kind), ("?", "?"))
        print(f"  {kind}: {name} @ {info[0]}:{info[1]}")
    print(f"\nTotal: {len(entry_points)} entry points\n")

    # 4. Dead code
    dead = find_dead_code(definitions, callers, entry_points)
    print("=== DEAD CODE (no callers, not entry points) ===\n")
    print("NOTE: This is a hint, not ground truth. Static analysis cannot trace")
    print("callbacks passed as arguments or framework-instantiated objects.\n")
    if dead:
        for name, kind, file, lineno in dead:
            print(f"  {kind}: {name} @ {file}:{lineno}")
        print(f"\nTotal: {len(dead)} unreachable definitions")
    else:
        print("  None found")


if __name__ == "__main__":
    main()