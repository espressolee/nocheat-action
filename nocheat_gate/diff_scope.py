"""Git diff hunk parsing and AST scope slicing for Python PRs."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator
import configparser
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tomllib

_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
_PATH_STATUS_RE = re.compile(rb'(?:[ADMT]|[MRC][0-9]{1,3})')


class GitInputError(ValueError):
    """Git did not provide supported, complete input for the requested comparison."""


@dataclass(frozen=True)
class GitPathChange:
    status: str
    rel_path: str
    old_rel_path: str | None = None


@dataclass(frozen=True)
class ChangedScope:
    """An AST function/class/module slice overlapping changed diff lines."""

    rel_path: str
    scope_name: str
    start_line: int
    end_line: int
    diff_lines: frozenset[int]
    is_fallback_callee: bool = False


@dataclass(frozen=True)
class FileDiffSummary:
    """Summary of one Python file modified between base and head."""

    rel_path: str
    status: str  # Git status letter: A, M, D, R...
    added_lines: frozenset[int]
    removed_lines: tuple[str, ...]
    has_deletions: bool
    old_rel_path: str | None = None

    @property
    def base_rel_path(self) -> str:
        return self.old_rel_path if self.status == 'R' and self.old_rel_path is not None else self.rel_path


@dataclass(frozen=True)
class DiffScope:
    """Diff-scoped production and test slices extracted from a PR commit range."""

    base_ref: str
    head_ref: str
    prod_files: tuple[FileDiffSummary, ...]
    test_files: tuple[FileDiffSummary, ...]
    prod_scopes: tuple[ChangedScope, ...]
    callee_fallback_scopes: tuple[ChangedScope, ...]
    pytest_python_files: tuple[str, ...] | None = None
    pytest_testpaths: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()


_TS_JS_EXTENSIONS = (".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs")
_DEFAULT_PYTEST_FILES = ('test_*.py', '*_test.py')


def matches_pytest_file(rel_path: str, patterns: tuple[str, ...], testpaths: tuple[str, ...] = ()) -> bool:
    path = PurePosixPath(rel_path)
    if testpaths and not any(
        fnmatchcase(candidate.as_posix(), pattern)
        for candidate in (path, *path.parents)
        for pattern in testpaths
    ):
        return False
    name = path.name
    return name.endswith('.py') and any(
        fnmatchcase(name, pattern) if '/' not in pattern else (
            fnmatchcase(rel_path, pattern) or fnmatchcase('/' + rel_path, '*/' + pattern)
        )
        for pattern in patterns
    )


def is_test_path(rel_path: str) -> bool:
    """Return True if `rel_path` matches common Python or TS/JS test path conventions."""
    pure = PurePosixPath(rel_path)
    parts = pure.parts
    name = pure.name
    lower_name = name.lower()
    if any(p in {"tests", "test", "__tests__"} for p in parts[:-1]):
        return True
    if name.endswith(".py"):
        return (
            name.startswith("test_")
            or name.endswith("_test.py")
            or name in {"test.py", "tests.py", "conftest.py"}
        )
    if lower_name.endswith(_TS_JS_EXTENSIONS):
        for ext in _TS_JS_EXTENSIONS:
            if lower_name.endswith(f".test{ext}") or lower_name.endswith(f".spec{ext}"):
                return True
        if lower_name.startswith("test."):
            return True
    return False


def _run_git(repo_dir: Path, args: list[str]) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_dir), "--literal-pathspecs", *args],
            check=True, capture_output=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise GitInputError('GIT_DIFF_UNAVAILABLE') from exc


def parse_git_path_changes(raw: bytes) -> tuple[GitPathChange, ...]:
    """Read Git's name-status -z stream without splitting or normalizing paths."""
    if not raw:
        return ()
    if not raw.endswith(b'\0'):
        raise GitInputError('GIT_DIFF_METADATA_INVALID: missing terminating NUL')
    fields = raw[:-1].split(b'\0')
    changes = []
    position = 0
    while position < len(fields):
        token = fields[position]
        row = len(changes) + 1
        if not _PATH_STATUS_RE.fullmatch(token) or (len(token) > 1 and int(token[1:]) > 100):
            raise GitInputError(f'GIT_DIFF_METADATA_INVALID: unsupported status in record {row}')
        status = chr(token[0])
        count = 2 if status in {'R', 'C'} else 1
        paths = fields[position + 1:position + 1 + count]
        if len(paths) != count or not all(paths):
            raise GitInputError(f'GIT_DIFF_METADATA_INVALID: missing path in record {row}')
        try:
            names = [path.decode('utf-8') for path in paths]
        except UnicodeDecodeError as exc:
            raise GitInputError(f'GIT_PATH_ENCODING_UNSUPPORTED: record {row}') from exc
        if any(PurePosixPath(name).is_absolute() or
               any(part in {'', '.', '..'} for part in name.split('/')) for name in names):
            raise GitInputError(f'GIT_DIFF_METADATA_INVALID: invalid path in record {row}')
        changes.append(GitPathChange(status, names[-1], names[0] if count == 2 else None))
        position += count + 1
    return tuple(changes)


def git_path_changes(repo_dir: Path, base: str, head: str) -> tuple[GitPathChange, ...]:
    return parse_git_path_changes(_run_git(
        repo_dir, ['diff', '--name-status', '-z', '--find-renames', '--find-copies', base, head, '--'],
    ))


def _source_text(raw: bytes, rel_path: str) -> str:
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise GitInputError(f'GIT_SOURCE_ENCODING_UNSUPPORTED: {rel_path!r}') from exc
    if '\0' in text:
        raise GitInputError(f'GIT_SOURCE_ENCODING_UNSUPPORTED: {rel_path!r} contains NUL')
    return text


def git_show_file(repo_dir: Path, rev: str, rel_path: str) -> str | None:
    """Read a UTF-8 file from a git revision, returning None if absent."""
    proc = subprocess.run(
        ["git", "-C", str(repo_dir), "show", f"{rev}:{rel_path}"],
        capture_output=True,
    )
    if proc.returncode != 0:
        return None
    return _source_text(proc.stdout, rel_path)


def parse_unified_diff(diff_text: str) -> tuple[frozenset[int], tuple[str, ...], bool]:
    """Parse `git diff -U0` into added 1-based line numbers and removed lines."""
    added: set[int] = set()
    removed: list[str] = []
    has_deletions = False
    cur_new_line: int | None = None

    for raw in diff_text.split('\n'):
        m = _HUNK_HEADER_RE.match(raw)
        if m:
            cur_new_line = int(m.group(1))
            continue
        if cur_new_line is None:
            continue
        if raw.startswith("+"):
            added.add(cur_new_line)
            cur_new_line += 1
        elif raw.startswith("-"):
            has_deletions = True
            removed.append(raw[1:])
    return frozenset(added), tuple(removed), has_deletions


@dataclass(frozen=True)
class _PytestCollection:
    patterns: tuple[str, ...] | None = None
    paths: tuple[str, ...] = ()
    standard_names: bool = True
    issue: str | None = None


def _pytest_collection(repo: Path, revision: str) -> _PytestCollection:
    """Read reviewed collection settings with pytest.ini precedence."""
    def args(name: str) -> tuple[str, ...] | None:
        if name not in filters:
            return None
        value = filters[name]
        if isinstance(value, str):
            return tuple(shlex.split(value))
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return tuple(value)
        raise ValueError(f'invalid {name}')

    filters: dict = {}
    try:
        for name in ('pytest.ini', '.pytest.ini'):
            raw = git_show_file(repo, revision, name)
            if raw is None:
                continue
            config = configparser.ConfigParser(interpolation=None)
            config.read_string(raw)
            if name == 'pytest.ini' or config.has_section('pytest'):
                filters = dict(config['pytest']) if config.has_section('pytest') else {}
                break
        else:
            raw = git_show_file(repo, revision, 'pyproject.toml')
            if raw is not None:
                pytest_config = tomllib.loads(raw).get('tool', {}).get('pytest', {})
                filters = pytest_config.get('ini_options', {})
        if not isinstance(filters, dict):
            raise ValueError('invalid pytest settings table')
        paths = tuple(PurePosixPath(path) for path in args('testpaths') or ())
        if any(path.is_absolute() or '..' in path.parts for path in paths):
            raise ValueError('testpaths must stay within the repository')
        standard_names = (
            args('python_functions') in (None, ('test',))
            and args('python_classes') in (None, ('Test',))
        )
        return _PytestCollection(args('python_files'), tuple(path.as_posix() for path in paths), standard_names)
    except (configparser.Error, UnicodeError, ValueError, AttributeError):
        return _PytestCollection(issue='PYTEST_COLLECTION_AMBIGUOUS: invalid base pytest settings')


_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _configured_test_role(
    repo: Path, base_ref: str, head_ref: str, rel_path: str, old_path: str, status: str,
    collection: _PytestCollection, *, conventional: bool = False,
) -> bool | None:
    """Use an explicit collection location and source role for nonstandard test paths.

    None means the source role cannot be established with the supported rules.
    """
    patterns = collection.patterns if collection.patterns is not None else _DEFAULT_PYTEST_FILES
    if not (collection.paths or conventional) or not matches_pytest_file(rel_path, patterns, collection.paths):
        return False
    if not collection.standard_names:
        return None

    found = False
    unsupported_role = False
    unreadable = False
    revisions = [(base_ref, old_path)] if status != 'A' else []
    if status != 'D':
        revisions.append((head_ref, rel_path))
    for revision, path in revisions:
        if not matches_pytest_file(path, patterns, collection.paths):
            continue
        try:
            source = git_show_file(repo, revision, path)
            if source is None:
                unreadable = True
                continue
            tree = ast.parse(source, filename=path)
        except (SyntaxError, UnicodeError):
            unreadable = True
            continue
        functions = list(tree.body)
        for item in tree.body:
            if isinstance(item, ast.ClassDef) and (
                item.name.startswith('Test')
                or any(isinstance(base, ast.Name) and base.id == 'TestCase'
                       or isinstance(base, ast.Attribute) and base.attr == 'TestCase' for base in item.bases)
            ):
                functions.extend(item.body)
                # A collectible subclass can inherit methods absent from this file.
                # The local soundness checker only inspects declared test bodies.
                unsupported_role |= bool(item.bases or item.keywords)
        names = [item.name for item in functions if isinstance(item, _FUNCTIONS)]
        found |= any(name.startswith('test_') for name in names)
        unsupported_role |= any(name.startswith('test') and not name.startswith('test_') for name in names)
    if unsupported_role:
        return None
    if found:
        # A repository-wide or wildcard path is insufficient to classify a source file as a test.
        if not conventional and not any(path != '.' and not any(c in path for c in '*?[')
                                        and matches_pytest_file(rel_path, patterns, (path,))
                                        for path in collection.paths):
            return None
        return True
    return None if unreadable else False


def ast_scopes_for_lines(
    rel_path: str,
    source: str,
    target_lines: frozenset[int],
    *,
    is_fallback_callee: bool = False,
) -> list[ChangedScope]:
    """Map changed line numbers to minimal enclosing AST functions or top blocks."""
    if not target_lines or not source.strip():
        return []
    try:
        tree = ast.parse(source, filename=rel_path)
    except SyntaxError:
        return []

    scopes: dict[tuple[str, int, int], set[int]] = {}

    def _walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}.{child.name}" if prefix else child.name
                start = child.lineno
                end = getattr(child, "end_lineno", start)
                overlap = {ln for ln in target_lines if start <= ln <= end}
                # Prefer innermost nested function if a nested def contains the change.
                nested_hit = any(
                    isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and sub is not child
                    and any(
                        sub.lineno <= ln <= getattr(sub, "end_lineno", sub.lineno)
                        for ln in overlap
                    )
                    for sub in ast.walk(child)
                )
                if overlap and not nested_hit:
                    scopes.setdefault((name, start, end), set()).update(overlap)
                else:
                    _walk(child, name)
            elif isinstance(child, ast.ClassDef):
                cname = f"{prefix}.{child.name}" if prefix else child.name
                _walk(child, cname)

    _walk(tree, "")

    # Capture module-level lines not belonging to any matched function/method.
    covered = {ln for lines in scopes.values() for ln in lines}
    uncovered = target_lines - covered
    if uncovered:
        for stmt in tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            start = stmt.lineno
            end = getattr(stmt, "end_lineno", start)
            overlap = {ln for ln in uncovered if start <= ln <= end}
            if overlap:
                scopes.setdefault(("<module>", start, end), set()).update(overlap)

    return [
        ChangedScope(
            rel_path=rel_path,
            scope_name=name,
            start_line=start,
            end_line=end,
            diff_lines=frozenset(lines),
            is_fallback_callee=is_fallback_callee,
        )
        for (name, start, end), lines in sorted(scopes.items(), key=lambda item: item[0][1])
    ]


def ambiguous_module_reason(module: str, paths: list[str]) -> str:
    return f"AMBIGUOUS_MODULE: {module} ({', '.join(sorted(paths))})"


def python_module_name(rel_path: str) -> str:
    """Name a source file in the analyzer's root and src layouts."""
    parts = PurePosixPath(rel_path).with_suffix('').parts
    if parts and parts[0] == 'src':
        parts = parts[1:]
    if parts and parts[-1] == '__init__':
        parts = parts[:-1]
    return '.'.join(parts)


def python_module_paths(module: str) -> tuple[str, ...]:
    stem = module.replace('.', '/')
    return (f'{stem}.py', f'src/{stem}.py', f'{stem}/__init__.py', f'src/{stem}/__init__.py')


def _resolve_module_to_relpath(repo_dir: Path, head_ref: str, module: str, issues: list[str]) -> str | None:
    """Resolve a static module only when the head tree has one matching source path."""
    paths = [path for path in python_module_paths(module) if git_show_file(repo_dir, head_ref, path) is not None]
    if len(paths) > 1:
        issues.append(ambiguous_module_reason(module, paths))
    return paths[0] if len(paths) == 1 else None


def _scope_nodes(statements: Iterable[ast.AST]) -> Iterator[ast.AST]:
    """Walk a single lexical body without attributing calls inside separate scopes."""
    for node in statements:
        yield node
        if not isinstance(node, (*_FUNCTIONS, ast.ClassDef, ast.Lambda, *_COMPREHENSIONS)):
            yield from _scope_nodes(ast.iter_child_nodes(node))


def _call_sites(
    nodes: Iterable[ast.AST], shadowed: frozenset[str] = frozenset(),
) -> Iterator[tuple[ast.Call, frozenset[str]]]:
    for node in nodes:
        if isinstance(node, ast.Call):
            yield node, shadowed
        elif isinstance(node, _COMPREHENSIONS):
            first, *rest = node.generators
            # Python evaluates the first iterable in the enclosing scope.
            yield from _call_sites(_scope_nodes((first.iter,)), shadowed)
            # All iteration targets are local to the comprehension, including
            # targets in later clauses that are still unbound at an earlier use.
            local_names = frozenset(
                part.id for generator in node.generators for part in ast.walk(generator.target)
                if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Store)
            )
            inside = list(first.ifs)
            for generator in rest:
                inside.append(generator.iter)
                inside.extend(generator.ifs)
            inside.extend((node.key, node.value) if isinstance(node, ast.DictComp) else (node.elt,))
            yield from _call_sites(_scope_nodes(inside), shadowed | local_names)


def _static_import_calls(
    statements: list[ast.stmt],
    inherited: dict[tuple[str, ...], tuple[str, str | None]] | None = None,
    parameters: Iterable[str] = (),
) -> Iterator[tuple[str, str]]:
    """Yield exact direct-import calls, skipping rebound or unsupported names."""
    nodes = list(_scope_nodes(statements))
    direct_imports = {node for node in statements if isinstance(node, (ast.Import, ast.ImportFrom))}
    bindings: dict[tuple[str, ...], tuple[str, str | None]] = {}
    namespaces: dict[str, tuple[str, str | None]] = {}
    first_binding_line: dict[tuple[str, ...], int] = {}
    hidden = set(parameters)
    for node in nodes:
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            hidden.add(node.id)
        elif isinstance(node, _COMPREHENSIONS):
            # Assignment expressions can rebind a name in the enclosing scope.
            hidden.update(part.target.id for part in ast.walk(node)
                          if isinstance(part, ast.NamedExpr) and isinstance(part.target, ast.Name))
        elif isinstance(node, (*_FUNCTIONS, ast.ClassDef)):
            hidden.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            hidden.update(node.names)
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
            hidden.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            hidden.add(node.rest)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == '*':
                    return
                if isinstance(node, ast.Import):
                    qualifier = tuple(alias.name.split('.')) if alias.asname is None else (alias.asname,)
                    target = (alias.name, None)
                    namespace = (alias.name if alias.asname else qualifier[0], None)
                else:
                    qualifier = (alias.asname or alias.name,)
                    target = (node.module or '', alias.name) if not node.level else ('', alias.name)
                    namespace = target
                if node not in direct_imports or not target[0] or namespaces.get(qualifier[0], namespace) != namespace:
                    hidden.add(qualifier[0])
                else:
                    namespaces[qualifier[0]] = namespace
                    bindings[qualifier] = target
                    first_binding_line.setdefault(qualifier, node.lineno)
    rebound = hidden | {key[0] for key in bindings}
    active = {key: value for key, value in (inherited or {}).items() if key[0] not in rebound}
    active.update((key, value) for key, value in bindings.items() if key[0] not in hidden)

    for node, shadowed in _call_sites(nodes):
        receiver = node.func
        suffix = []
        while isinstance(receiver, ast.Attribute):
            suffix.append(receiver.attr)
            receiver = receiver.value
        if not isinstance(receiver, ast.Name) or receiver.id in shadowed:
            continue
        qualifier = (receiver.id, *reversed(suffix))
        key = qualifier if len(qualifier) == 1 else qualifier[:-1]
        if node.lineno <= first_binding_line.get(key, 0):
            continue
        module, symbol = active.get(key, (None, None))
        if len(qualifier) == 1:
            if module and symbol:
                yield module, symbol
        elif module:
            yield (f'{module}.{symbol}' if symbol else module), qualifier[-1]

    children = [item for item in statements if isinstance(item, _FUNCTIONS)]
    for item in statements:
        if isinstance(item, ast.ClassDef):
            children.extend(method for method in item.body if isinstance(method, _FUNCTIONS))
    for child in children:
        arguments = [node.arg for node in ast.iter_child_nodes(child.args) if isinstance(node, ast.arg)]
        yield from _static_import_calls(child.body, active, arguments)


def resolve_callee_scopes_from_tests(
    repo_dir: Path,
    head_ref: str,
    test_files: tuple[FileDiffSummary, ...],
    *,
    issues: list[str] | None = None,
) -> tuple[ChangedScope, ...]:
    """Resolve static calls in changed tests to the named functions in the head tree."""
    wanted: dict[str, set[str]] = {}
    issues = [] if issues is None else issues
    test_paths = {test.rel_path for test in test_files}
    for test in test_files:
        if test.status == 'D':
            continue
        source = git_show_file(repo_dir, head_ref, test.rel_path)
        if not source:
            continue
        try:
            tree = ast.parse(source, filename=test.rel_path)
        except SyntaxError:
            continue
        for module, symbol in _static_import_calls(tree.body):
            wanted.setdefault(module, set()).add(symbol)

    scopes: list[ChangedScope] = []
    for module, names in sorted(wanted.items()):
        path = _resolve_module_to_relpath(repo_dir, head_ref, module, issues)
        if path is None or path in test_paths or is_test_path(path):
            continue
        source = git_show_file(repo_dir, head_ref, path)
        if not source:
            continue
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError:
            continue
        lines = frozenset(
            line for item in tree.body if isinstance(item, (*_FUNCTIONS, ast.ClassDef)) and item.name in names
            for line in range(item.lineno, item.end_lineno + 1)
        )
        scopes.extend(ast_scopes_for_lines(path, source, lines, is_fallback_callee=True))
    return tuple(sorted(scopes, key=lambda scope: (scope.rel_path, scope.start_line)))


def _is_supported_source(rel_path: str) -> bool:
    parts = PurePosixPath(rel_path).parts
    if parts and parts[0] in {"fixtures", "testdata", "__fixtures__"}:
        return False
    lower = rel_path.lower()
    if lower.endswith(".py"):
        return True
    if lower.endswith(".d.ts"):
        return False
    return lower.endswith(_TS_JS_EXTENSIONS)


def _source_changes(
    repo: Path, base: str, head: str, changes: Iterable[GitPathChange],
    collection: _PytestCollection, issues: list[str],
) -> Iterator[tuple[GitPathChange, bool]]:
    """Project Git records into source/test changes under the reviewed base collection rules."""
    patterns = collection.patterns if collection.patterns is not None else _DEFAULT_PYTEST_FILES

    def test_role(change: GitPathChange, *, check_body: bool = False) -> bool | None:
        path = change.rel_path
        conventional = is_test_path(path)
        if path.endswith('.py'):
            if collection.issue:
                issues.append(collection.issue)
            if PurePosixPath(path).name == 'conftest.py':
                return True
            if path == 'test.py' and not collection.paths and collection.patterns is None:
                return True  # The runner also supports the standalone test script.
            if not matches_pytest_file(path, patterns, collection.paths):
                return False
            if conventional and not check_body:
                return True
        elif conventional:
            return True
        status = 'A' if change.status == 'C' else change.status
        role = _configured_test_role(repo, base, head, path, path, status, collection,
                                     conventional=conventional)
        if role is None:
            issues.append(f'PYTEST_COLLECTION_AMBIGUOUS: cannot establish test role for {path}')
        return role

    for change in changes:
        new_source = _is_supported_source(change.rel_path)
        if change.status == 'R' and change.old_rel_path is not None:
            old_source = _is_supported_source(change.old_rel_path)
            removed = GitPathChange('D', change.old_rel_path)
            added = GitPathChange('A', change.rel_path, change.old_rel_path)
            old_role = test_role(removed, check_body=True) if old_source else False
            new_role = test_role(added, check_body=True) if new_source else False
            if old_role is None or new_role is None:
                continue
            if old_source != new_source or old_role != new_role:
                if old_source:
                    yield removed, old_role
                if new_source:
                    yield added, new_role
            elif new_source:
                yield change, new_role
        elif new_source:
            role = test_role(change)
            if role is not None:
                yield change, role


def _file_diff(
    repo: Path, base: str, head: str, change: GitPathChange,
) -> tuple[FileDiffSummary, str | None]:
    path = change.rel_path
    old_path = change.old_rel_path if change.status == 'R' and change.old_rel_path is not None else path
    # An unchanged rename can contain unsupported bytes without any textual hunks.
    if change.status not in {'A', 'C'} and git_show_file(repo, base, old_path) is None:
        raise GitInputError(f'GIT_SOURCE_UNAVAILABLE: {old_path!r} at base')
    source = None if change.status == 'D' else git_show_file(repo, head, path)
    if change.status != 'D' and source is None:
        raise GitInputError(f'GIT_SOURCE_UNAVAILABLE: {path!r} at head')

    flags = ['diff', '--unified=0', '--no-ext-diff', '--no-textconv', '--text']
    if change.status == 'R':
        argv = [*flags, f'{base}:{old_path}', f'{head}:{path}']
    else:
        # Copy destinations introduce source at a new path and get addition hunks.
        argv = [*flags, '--no-renames', base, head, '--', path]
    added, removed, has_deletions = parse_unified_diff(_source_text(_run_git(repo, argv), path))
    return FileDiffSummary(path, change.status, added, removed, has_deletions, change.old_rel_path), source


def extract_diff_scope(repo_dir: Path, base_ref: str, head_ref: str) -> DiffScope:
    """Extract production and test diff slices from `base_ref..head_ref`."""
    changes = git_path_changes(repo_dir, base_ref, head_ref)
    prod_files: list[FileDiffSummary] = []
    test_files: list[FileDiffSummary] = []
    collection = _pytest_collection(repo_dir, base_ref)
    issues: list[str] = []
    head_sources = {}

    for change, test_path in _source_changes(repo_dir, base_ref, head_ref, changes, collection, issues):
        summary, source = _file_diff(repo_dir, base_ref, head_ref, change)
        rel_path = summary.rel_path
        if source is not None:
            head_sources[rel_path] = source
        if test_path:
            test_files.append(summary)
        else:
            prod_files.append(summary)

    prod_scopes: list[ChangedScope] = []
    for pf in prod_files:
        if pf.status == "D" or not pf.added_lines or not pf.rel_path.endswith(".py"):
            continue
        src = head_sources.get(pf.rel_path)
        if src:
            prod_scopes.extend(ast_scopes_for_lines(pf.rel_path, src, pf.added_lines))

    py_tests = tuple(tf for tf in test_files if tf.rel_path.endswith(".py"))
    callee_scopes: tuple[ChangedScope, ...] = ()
    if not prod_scopes and py_tests:
        callee_scopes = resolve_callee_scopes_from_tests(
            repo_dir, head_ref, py_tests, issues=issues,
        )

    return DiffScope(
        base_ref=base_ref,
        head_ref=head_ref,
        prod_files=tuple(prod_files),
        test_files=tuple(test_files),
        prod_scopes=tuple(prod_scopes),
        callee_fallback_scopes=callee_scopes,
        pytest_python_files=collection.patterns,
        pytest_testpaths=collection.paths,
        issues=tuple(dict.fromkeys(issues)),
    )
