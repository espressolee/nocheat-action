"""Explicit Node/parser runtime locations, independent of benchmark checkouts."""

import os
from pathlib import Path

# Editable/source installs use the project root. Wheel installs can point explicitly
# at a prepared runtime directory without borrowing anything from a target checkout.
RUNTIME_ROOT = Path(os.environ.get('NOCHEAT_RUNTIME_ROOT', str(Path(__file__).resolve().parent.parent))).resolve()
NODE_MODULES = RUNTIME_ROOT / 'node_modules'
WASM_DIR = RUNTIME_ROOT / '.runtime' / 'wasm'
WASM_FILES = ('tree-sitter.cjs', 'tree-sitter.wasm', 'tree-sitter-typescript.wasm',
              'tree-sitter-tsx.wasm', 'tree-sitter-javascript.wasm')
