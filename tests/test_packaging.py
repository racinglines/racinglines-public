"""The installed package: --version, the missing-package message, and the configs a wheel bundles (setup.py)."""

import ast
from pathlib import Path

import pytest

from racinglines import cli, paths

REPO = Path(__file__).resolve().parents[1]


def test_version(capsys):
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.startswith("racinglines ")


def test_missing_package_names_the_pip_install(monkeypatch, capsys):
    def boom(argv):
        raise ModuleNotFoundError("No module named 'sklearn'", name="sklearn")
    monkeypatch.setattr(cli, "_run", boom)
    assert cli.main(["model", "forecast", "f1"]) == 2
    err = capsys.readouterr().err
    assert "needs sklearn" in err and "pip install scikit-learn" in err


def test_missing_racinglines_module_still_raises(monkeypatch):
    def boom(argv):
        raise ModuleNotFoundError("No module named 'racinglines.nope'", name="racinglines.nope")
    monkeypatch.setattr(cli, "_run", boom)
    with pytest.raises(ModuleNotFoundError):
        cli.main(["model", "forecast", "f1"])


def test_checkout_paths_unchanged():
    assert paths.CHECKOUT and paths.ROOT == REPO and paths.HOME == REPO


def test_wheel_bundles_every_config_dir_the_package_reads():
    """Each `ROOT / "<dir>"` the package reads at run time is a bundled dir in setup.py (or a checkout-only one)."""
    tree = ast.parse((REPO / "setup.py").read_text())
    bundled = next(ast.literal_eval(n.value) for n in tree.body
                   if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "BUNDLED")
    checkout_only = {"data", "reports", "alembic.ini", "site", "pitch.html", "live"}   # server/ops paths
    read = set()
    for f in (REPO / "racinglines").rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and isinstance(node.right, ast.Constant)
                    and isinstance(node.left, (ast.Name, ast.Attribute))
                    and getattr(node.left, "id", getattr(node.left, "attr", "")) == "ROOT"):
                read.add(str(node.right.value).split("/")[0])
    assert read - checkout_only <= set(bundled), f"bundle these in setup.py: {sorted(read - checkout_only - set(bundled))}"


def test_cli_source_leaves_paths_unimported():
    """use_demo sets RACINGLINES_DATA before racinglines.paths is first imported; an early import fixes the wrong DATA."""
    import subprocess
    import sys
    code = "import sys, racinglines.cli.source as s, racinglines.paths as p; print(s.HOME == p.HOME)"
    pre = "import sys, racinglines.cli.source; print('racinglines.paths' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", pre], capture_output=True, text=True, cwd=REPO).stdout.strip() == "False"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO).stdout.strip() == "True"
