"""Loader for the per-case study modules.

The case directories are named with hyphens (``cases/novel-baswedan``) so they
read well in a repository browser, which means they cannot be imported as
normal dotted packages. This helper loads a module file by path instead.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from .io import REPO_ROOT

CASES_DIR = REPO_ROOT / "cases"


def load_case_module(case: str, module_file: str) -> ModuleType:
    """Import ``cases/<case>/<module_file>`` and return the module object."""
    path = CASES_DIR / case / module_file
    if not path.exists():
        raise FileNotFoundError(f"No such case module: {path}")
    mod_name = f"idtweets_case_{case.replace('-', '_')}_{path.stem}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module
