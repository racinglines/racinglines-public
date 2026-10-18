"""The wheel's file list (the rest of the metadata is in pyproject.toml).

sports/, markets/, exchanges/ and migrations/ stay at the repository root, where the VM, the scripts and the docs read them, and a
wheel bundles copies as racinglines/_bundled/<dir>/, which racinglines/paths.py reads when there is no checkout."""

from pathlib import Path

from setuptools import find_packages, setup

BUNDLED = ("sports", "markets", "exchanges", "migrations")
bundled = {"racinglines._bundled." + ".".join(d.parts): d.as_posix()
           for top in BUNDLED for d in [Path(top), *sorted(p for p in Path(top).rglob("*")
                                                         if p.is_dir() and p.name != "__pycache__")]}

setup(
    packages=find_packages(include=["racinglines*"]) + list(bundled),
    package_dir=bundled,
    package_data={
        "racinglines.web": ["templates/*.html", "static/*", "static/brand/*", "term/*"],
        "racinglines.weather": ["*.toml"],
        "racinglines.books": ["*.css"],
        "racinglines.tui": ["*.tcss"],
        "racinglines.demo": ["packs/*/*.toml", "packs/*/files/**/*"],
        **{name: ["*.toml", "*.mako"] for name in bundled},
    },
)
