# Changelog

Every version on PyPI, newest first. Each one is a single commit and tag `vX` in this repository
(<https://github.com/racinglines/racinglines-public>), and that is the tree PyPI's build came from.

## 1.0.1 (2026-10-19)

- Published from this repository: PyPI's provenance for 1.0.1 names `racinglines/racinglines-public` and the tag `v1.0.1`.
- The web app's GitHub links (the nav and the footer) point at this repository.
- The racinglines 101 page links its two docs pages (F1 calibration, F1 evaluation) at <https://racinglines.bet/docs>
  instead of relative paths that led nowhere in an installed package.
- README: the from-source steps clone this repository and `pip install -e ".[web]"`.

## 1.0.0 (2026-10-18)

First release.

- The F1 pricing engine and the `racinglines` CLI, the UCI downhill sim and the MK8DX (kart) demo.
- `pip install "racinglines[web]"`: the web app (Markets, Trade, History, Lab, the live viewer) with F$ fantasy trading,
  its database schema and migrations.
- `racinglines web --demo` creates the demo accounts (Try the demo signs in as `maker`); `/` is the sign-in page.
