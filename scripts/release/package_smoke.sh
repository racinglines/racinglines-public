#!/usr/bin/env bash
# The package as `pip install racinglines` gets it: build the sdist and wheel, check their metadata, install the wheel
# with only its core dependencies into a fresh venv, and run the CLI from an empty folder with no checkout, database or
# network. Both release workflows (deploy/ci/release.yml, oss/overlay/.github/workflows/release.yml) run it before any
# push or upload; run it locally before a tag.
#
#   bash scripts/release/package_smoke.sh [DIST_DIR]      # DIST_DIR default: dist/ (rebuilt)
#   RELEASE=1 bash scripts/release/package_smoke.sh ...   # before an upload: the LICENSE must be the plain MIT text
#   EXPECT_VERSION=0.1.0rc1 bash ...                      # fail unless the built package says this version
set -euo pipefail
export PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
PY=${PYTHON:-python3}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
DIST=${1:-$REPO/dist}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "smoke: building into $DIST"
rm -rf "$DIST" "$REPO/build"
"$PY" -m venv "$WORK/tools"                     # build and twine in their own venv, not the caller's
"$WORK/tools/bin/pip" install --quiet --upgrade pip build twine
"$WORK/tools/bin/python" -m build --outdir "$DIST" "$REPO" > "$WORK/build.log" 2>&1 || { tail -40 "$WORK/build.log"; exit 1; }
"$WORK/tools/bin/python" -m twine check --strict "$DIST"/*

wheel=$(ls "$DIST"/*.whl)
echo "smoke: bundled configs in $(basename "$wheel")"
"$PY" - "$wheel" <<'PY'
import sys, zipfile
names = set(zipfile.ZipFile(sys.argv[1]).namelist())
need = ["racinglines/_bundled/sports/f1.toml", "racinglines/_bundled/markets/kinds.toml",
        "racinglines/_bundled/exchanges/og.toml"]
missing = [n for n in need if n not in names]
leaked = sorted(n for n in names if n.split("/")[0] in ("tests", "data", "docs", "scripts", "deploy", "reports", "paper"))
if missing or leaked:
    sys.exit(f"wheel: missing {missing}, files that must not ship {leaked[:10]}")
print(f"  {len(names)} files, the configs present, no tests/data/docs/scripts")
PY
"$PY" - "$wheel" "${RELEASE:-0}" <<'PY'
import email, re, sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
meta = email.message_from_bytes(z.read(next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))))
text = z.read(next(n for n in z.namelist() if n.endswith(".dist-info/licenses/LICENSE"))).decode()
words = " ".join(text.split())
plain = (meta["License-Expression"] == "MIT" and words.startswith("MIT License Copyright (c)")
         and "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software" in words
         and not re.search(r"commercial|additional restriction", words, re.I))   # plain MIT itself says "without restriction"
print(f"  licence: {meta['License-Expression']}, LICENSE {'is' if plain else 'is NOT'} the plain MIT text")
if not plain and sys.argv[2] == "1":
    sys.exit("licence: a release ships the plain MIT LICENSE only")
PY

echo "smoke: fresh venv, core dependencies only"
"$PY" -m venv "$WORK/venv"
"$WORK/venv/bin/pip" install --quiet "$wheel"
mkdir -p "$WORK/empty" "$WORK/home"
cd "$WORK/empty"
run() { env -u DATABASE_URL -u RACINGLINES_DATA HOME="$WORK/home" RACINGLINES_PROGRESS_SEC=0 "$WORK/venv/bin/racinglines" "$@"; }
run --version
if [ -n "${EXPECT_VERSION:-}" ] && [ "$(run --version)" != "racinglines $EXPECT_VERSION" ]; then
  echo "smoke: built $(run --version), wanted $EXPECT_VERSION"; exit 1; fi
run --help > /dev/null
run check --offline --no-db --sport f1
if [ -n "$(ls -A "$WORK/empty")" ]; then echo "smoke: the CLI wrote into the working folder:"; ls -A "$WORK/empty"; exit 1; fi
site=$("$WORK/venv/bin/python" -c 'import racinglines, pathlib; print(pathlib.Path(racinglines.__file__).parent)')
if [ -e "$site/data" ] || [ -e "$site/../data" ]; then echo "smoke: the CLI wrote data/ into site-packages"; exit 1; fi
echo "smoke: passed"
