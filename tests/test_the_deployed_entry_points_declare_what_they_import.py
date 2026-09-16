"""
Two things run this code away from this machine, and they install different files.

`.github/workflows/forecast.yml` installs `requirements.txt` and runs
`pipelines/forecast.py`. Streamlit Community Cloud installs the same file and
runs `app/streamlit_app.py`. Everything else - the tests, the training pipeline,
the evidence rebuild - gets `requirements-dev.txt`, which is a superset.

So a package used on a live path but declared only for development works
everywhere except where it matters, and fails on the first scheduled run rather
than in CI. That is exactly what happened: `features.py` carries the stations
onto the grid with `sklearn.neighbors.BallTree`, scikit-learn sat in
requirements-dev.txt, the test suite passed on every push, and the first live
forecast died on the import in under a second.

This walks the imports of both deployed entry points - through this repository's
own modules, which is where the offending import was - and holds every
third-party package it reaches to what `requirements.txt` declares.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / "requirements.txt"
DEV = ROOT / "requirements-dev.txt"

#: The two things that run this code somewhere other than here.
ENTRY_POINTS = {
    "the scheduled forecast (forecast.yml)": ROOT / "pipelines" / "forecast.py",
    "the Streamlit app (Community Cloud)": ROOT / "app" / "streamlit_app.py",
}

#: Import name -> distribution name, where they differ.
DISTRIBUTION = {"sklearn": "scikit-learn"}

#: Where this repository's own modules live. A local import is followed, not declared.
LOCAL_ROOTS = [ROOT / "src", ROOT / "app", ROOT / "pipelines", ROOT / "app" / "views"]

STDLIB = {
    "__future__", "argparse", "collections", "concurrent", "contextlib", "csv",
    "dataclasses", "datetime", "functools", "io", "itertools", "json", "math",
    "os", "pathlib", "random", "re", "shutil", "subprocess", "sys", "tempfile",
    "textwrap", "threading", "time", "typing", "urllib", "warnings", "zipfile",
}


def declared(path: Path) -> set[str]:
    """Distribution names pinned in a requirements file, lowercased."""
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        names.add(re.split(r"[=<>!~\[]", line, 1)[0].strip().lower())
    return names


def local_module(name: str) -> Path | None:
    """The file behind an import of this repository's own code, if that is what it is."""
    relative = Path(*name.split("."))
    for root in LOCAL_ROOTS:
        for candidate in (root / relative.with_suffix(".py"), root / relative / "__init__.py"):
            if candidate.is_file():
                return candidate
    return None


def page_files(tree: ast.AST, folder: Path) -> list[Path]:
    """The pages a Streamlit entry script names.

    `st.Page("views/forecast.py")` is a string, not an import, so an import crawl
    alone walks one file and proves nothing about the app. The pages are the app.
    """
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and node.value.endswith(".py"):
            candidate = folder / node.value
            if candidate.is_file():
                out.append(candidate)
    return out


def reachable_imports(entry: Path) -> tuple[set[str], set[Path]]:
    """Third-party top-level imports reachable from `entry`, following local modules.

    Imports inside functions count: the app reaches the forecast builder through one
    (`from wildfire.forecast import build`), which is precisely how the missing
    package stayed invisible until the page that calls it ran.
    """
    third_party: set[str] = set()
    seen: set[Path] = set()
    queue = [entry]
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        queue.extend(page_files(tree, path.parent))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                # `from wildfire import features` names a module, not an attribute of
                # one, and following only `wildfire` stops at its __init__ - which is
                # how the crawl first walked straight past the file that needed
                # scikit-learn. Try each name as a submodule, and keep the package
                # itself for the cases where it really is an attribute.
                names = [f"{node.module}.{alias.name}" for alias in node.names] + [node.module]
            else:
                continue
            for name in names:
                target = local_module(name)
                if target is not None:
                    queue.append(target)
                    continue
                top = name.split(".")[0]
                if top not in STDLIB and local_module(top) is None:
                    third_party.add(top)
    return third_party, seen


@pytest.mark.parametrize("label", sorted(ENTRY_POINTS))
def test_the_entry_point_only_imports_what_requirements_declares(label):
    entry = ENTRY_POINTS[label]
    packages, walked = reachable_imports(entry)
    assert len(walked) > 3, f"{label}: only {len(walked)} module(s) walked - the crawl found nothing"
    runtime = declared(RUNTIME)
    missing = sorted(DISTRIBUTION.get(p, p) for p in packages
                     if DISTRIBUTION.get(p, p).lower() not in runtime)
    assert not missing, (
        f"{label} imports {missing}, which requirements.txt does not declare. "
        "The suite installs requirements-dev.txt, so this passes CI and fails the "
        "moment the job or the app actually runs.")


def test_the_development_file_builds_on_the_runtime_one():
    """Otherwise the two lists drift apart and this check tests the wrong file."""
    assert "-r requirements.txt" in DEV.read_text(encoding="utf-8")


def test_scikit_learn_is_a_runtime_dependency():
    """Named directly, because the crawl above would also be satisfied by deleting
    the import that needs it - and the interpolation it performs is not optional."""
    assert "scikit-learn" in declared(RUNTIME)
    assert "BallTree" in (ROOT / "src" / "wildfire" / "features.py").read_text(encoding="utf-8")
