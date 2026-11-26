"""Bounded static Python reference/call impact; uncertainty selects all tests."""
from __future__ import annotations
import argparse
import ast
from collections.abc import Iterable
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import builtins
from itertools import islice
import os

from .diff_scope import ambiguous_module_reason, matches_pytest_file, python_module_name, python_module_paths


@dataclass(frozen=True)
class ImpactReport:
    changed: tuple[str, ...]
    impacted: tuple[str, ...]
    tests: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    fallback: bool
    issues: tuple[str, ...]
    candidates: tuple[str, ...] = ()


def _python_paths(root: Path):
    ignored = {'node_modules', 'out', 'build', 'dist', 'fixtures', 'testdata', '__pycache__'}
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in ignored)
        for name in sorted(files):
            if name.endswith('.py') and not name.startswith('.'):
                yield Path(directory) / name


def source_module_collisions(root: Path, source_paths: Iterable[str]) -> tuple[str, ...]:
    """Check known changed or fallback source identities before either runner starts."""
    modules = {python_module_name(path) for path in source_paths if path.endswith('.py')}
    issues = []
    for name in sorted(modules - {''}):
        paths = [path for path in python_module_paths(name) if (root / path).is_file()]
        if len(paths) > 1:
            issues.append(ambiguous_module_reason(name, paths))
    return tuple(issues)


def analyze_impact(root: Path, prod_files, scopes=(), *, pytest_python_files=None, pytest_testpaths=()) -> ImpactReport:
    root=root.resolve(); modules={}; trees={}; definitions={}; bindings={}; globals_by_file={}; issues=[]; edges=set()
    paths = list(islice(_python_paths(root), 513))
    if len(paths) > 512:
        return ImpactReport((), (), (), (), True, ('MODULE_LIMIT',))
    def pytest_file(p):
        if pytest_python_files is not None:
            rel = p.relative_to(root).as_posix()
            return matches_pytest_file(rel, pytest_python_files, pytest_testpaths)
        return p.name.startswith('test_') or p.name.endswith('_test.py') or p.name in {'test.py','tests.py'}
    candidates=tuple(str(p.relative_to(root)) for p in paths if pytest_file(p))
    def node(rel,name):return rel+':'+name
    for p in paths:
        rel=p.relative_to(root).as_posix()
        if p.is_symlink() or p.stat().st_size>131072:
            issues.append('SOURCE_UNAVAILABLE:'+rel);continue
        module = python_module_name(rel)
        if module in modules:
            reason = ambiguous_module_reason(module, [modules[module], rel])
            return ImpactReport((), (), (), (), True, (reason,), candidates)
        modules[module]=rel
        try: trees[rel]=ast.parse(p.read_text(),filename=rel)
        except (SyntaxError,UnicodeError,OSError):issues.append('PARSE_UNAVAILABLE:'+rel)
    reverse_modules={rel:mod for mod,rel in modules.items()}
    for rel,tree in trees.items():
        table={}; definitions[rel]={'<module>':tree}; bindings[rel]=table
        globals_by_file[rel]={n.id for statement in tree.body if isinstance(statement,(ast.Assign,ast.AnnAssign,ast.AugAssign))
                              for n in ast.walk(statement) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
        module=reverse_modules[rel]; package=module.split('.') if Path(rel).name=='__init__.py' else module.split('.')[:-1]
        for statement in tree.body:
            if isinstance(statement,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                if statement.decorator_list:issues.append('DECORATED_SCOPE:'+rel+':'+statement.name)
                definitions[rel][statement.name]=statement
                if isinstance(statement,ast.ClassDef):
                    if statement.bases:issues.append('INHERITANCE:'+rel+':'+statement.name)
                    for child in statement.body:
                        if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef)):
                            if child.decorator_list:issues.append('DECORATED_SCOPE:'+rel+':'+statement.name+'.'+child.name)
                            definitions[rel][statement.name+'.'+child.name]=child
                        elif not isinstance(child,ast.Pass) and not (isinstance(child,ast.Expr) and isinstance(child.value,ast.Constant) and isinstance(child.value.value,str)):
                            issues.append('CLASS_STATE:'+rel+':'+statement.name)
            elif isinstance(statement,ast.Import):
                for alias in statement.names:table[alias.asname or alias.name.split('.')[0]]=(alias.name if alias.asname else alias.name.split('.')[0],None)
            elif isinstance(statement,ast.ImportFrom):
                prefix=package[:len(package)-statement.level+1] if statement.level else []
                imported='.'.join(prefix+((statement.module or '').split('.') if statement.module else []))
                for alias in statement.names:
                    if alias.name=='*':issues.append('STAR_IMPORT:'+rel)
                    else:table[alias.asname or alias.name]=(imported,alias.name)
    def resolve(module,symbol,seen=None):
        seen=set() if seen is None else seen
        key=(module,symbol)
        if key in seen:return set()
        seen.add(key)
        rel=modules.get(module)
        if rel is None:return set()
        if not symbol:return {node(rel,'<module>')}
        if symbol in definitions.get(rel,{}):return {node(rel,symbol)}
        first,*tail=symbol.split('.')
        bound=bindings[rel].get(first)
        if bound:
            mod,sym=bound
            return resolve(mod,'.'.join(([sym] if sym else [])+tail) or None,seen)
        if module+'.'+symbol in modules:return resolve(module+'.'+symbol,None,seen)
        if symbol not in globals_by_file.get(rel,set()):issues.append('UNRESOLVED_LOCAL_SYMBOL:'+rel+':'+symbol)
        # A local data reference depends on module execution, not a guessed function.
        return {node(rel,'<module>')}
    def dotted(value):
        if isinstance(value,ast.Name):return [value.id]
        if isinstance(value,ast.Attribute):
            prefix=dotted(value.value)
            return prefix+[value.attr] if prefix else []
        return []
    def descendants(value):
        for child in ast.iter_child_nodes(value):
            if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                continue
            yield child
            yield from descendants(child)
    for rel,defs in definitions.items():
        mod=reverse_modules[rel]
        for name,definition in defs.items():
            caller=node(rel,name)
            if isinstance(definition,ast.ClassDef):
                for member in defs:
                    if member.startswith(name+'.'):edges.add((caller,node(rel,member)))
                continue
            rebound={n.id for n in descendants(definition) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
            if isinstance(definition,(ast.FunctionDef,ast.AsyncFunctionDef)):
                rebound.update(arg.arg for arg in ast.walk(definition.args) if isinstance(arg,ast.arg))
            for child in descendants(definition):
                if isinstance(child,ast.Call):
                    parts=dotted(child.func);targets=set()
                    if not parts:
                        issues.append('DYNAMIC_CALL:'+caller);continue
                    first=parts[0]
                    if first in rebound and first in bindings[rel]:issues.append('REBOUND_IMPORT:'+caller)
                    if first in {'getattr','setattr','eval','exec','__import__'} or parts[-1]=='import_module':
                        issues.append('DYNAMIC_CALL:'+caller);continue
                    if first=='self' and '.' in name:
                        targets=resolve(mod,name.rsplit('.',1)[0]+'.'+'.'.join(parts[1:]))
                        if targets=={node(rel,'<module>')}:issues.append('UNRESOLVED_METHOD:'+caller)
                    elif first in bindings[rel]:
                        target_mod,sym=bindings[rel][first]
                        suffix=([sym] if sym else [])+parts[1:]
                        while suffix and target_mod+'.'+suffix[0] in modules:
                            target_mod+='.'+suffix.pop(0)
                        targets=resolve(target_mod,'.'.join(suffix) or None)
                    elif len(parts)==1 and first in defs:targets=resolve(mod,first)
                    elif first not in vars(builtins):issues.append('UNRESOLVED_CALL:'+caller)
                    edges.update((caller,target) for target in targets)
                elif isinstance(child,(ast.Import,ast.ImportFrom)) and name!='<module>':
                    issues.append('FUNCTION_LOCAL_IMPORT:'+caller)
                elif isinstance(child,ast.Name) and isinstance(child.ctx,ast.Load):
                    if child.id in bindings[rel]:
                        target_mod,sym=bindings[rel][child.id]
                        edges.update((caller,target) for target in resolve(target_mod,sym))
                    elif child.id in defs:
                        edges.add((caller,node(rel,child.id)))
                # Every definition also depends on module initialization/constants.
            if name!='<module>':edges.add((caller,node(rel,'<module>')))
    changed=set()
    named={}
    for scope in scopes:named.setdefault(scope.rel_path,set()).add(scope.scope_name)
    for file in prod_files:
        if not file.rel_path.endswith('.py'):continue
        if file.status=='D':issues.append('DELETED_SOURCE:'+file.rel_path)
        names=named.get(file.rel_path)
        if not names or '<module>' in names:
            changed.update(node(file.rel_path,n) for n in definitions.get(file.rel_path,{'<module>':None}))
        else:changed.update(node(file.rel_path,n) for n in names)
    impacted=set(changed)
    while True:
        extra={caller for caller,callee in edges if callee in impacted}-impacted
        if not extra:break
        impacted.update(extra)
    fallback=bool(issues)
    if any(key.split(':',1)[0].endswith('conftest.py') for key in impacted):
        issues.append('IMPLICIT_PYTEST_FIXTURE');fallback=True
    tests=set(candidates) if fallback else {key.split(':',1)[0] for key in impacted if key.split(':',1)[0] in candidates}
    return ImpactReport(tuple(sorted(changed)),tuple(sorted(impacted)),tuple(sorted(tests)),tuple(sorted(edges)),fallback,tuple(sorted(set(issues))),candidates)


def main():
    from .diff_scope import extract_diff_scope
    from .runner import temporary_checkout
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--base',required=True);p.add_argument('--head',required=True);a=p.parse_args()
    scope=extract_diff_scope(a.repo,a.base,a.head)
    with temporary_checkout(a.repo,a.head) as root:
        report = analyze_impact(root, scope.prod_files, scope.prod_scopes,
                                pytest_python_files=scope.pytest_python_files, pytest_testpaths=scope.pytest_testpaths)
    payload = asdict(report)
    payload.pop('candidates')
    print(json.dumps(payload, indent=2))
if __name__=='__main__':main()
