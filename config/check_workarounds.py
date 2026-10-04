# lsqfitgp/config/check_workarounds.py
#
# Copyright (c) 2026, Giacomo Petrillo
#
# This file is part of lsqfitgp.
#
# lsqfitgp is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# lsqfitgp is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with lsqfitgp.  If not, see <http://www.gnu.org/licenses/>.

"""Report obsolete `WORKAROUND(pkg<ver)` / `WORKAROUND(pkg<=ver)` markers.

A marker is obsolete given the current floors in pyproject.toml, plus the floor
for `uv` (= the version pinned in .pre-commit-config.yaml).

Marker grammar:
    WORKAROUND(<pkg><op><version>): <free-text>
with <op> in {<, <=}. A marker is obsolete iff every supported version of
<pkg> (i.e., versions >= the floor) satisfies NOT (version <op> <version>).
The floor for dependencies and `python` comes from pyproject.toml. Markers that
do not fit the grammar are reported as malformed rather than silently ignored.
"""

import re
import subprocess
import sys
from pathlib import Path

import tomli
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

# Loose pattern: find any `WORKAROUND(...)` candidate to inspect.
CANDIDATE_RE = re.compile(r'WORKAROUND\(([^)]*)\)')
# Strict grammar applied to the inner contents.
INNER_RE = re.compile(r'\s*([A-Za-z0-9_.\-]+)\s*(<=|<)\s*(\S+)\s*')
# The uv version, in the `# frozen: <version>` comment on the rev line right
# below the uv-pre-commit repo line; also the uv installed by CI.
UV_REV_RE = re.compile(r'astral-sh/uv-pre-commit\s*\n.*#\s*frozen:\s*v?(\S+)')


def floors_from_pyproject(path: Path) -> dict[str, Version]:
    """Return {normalized_pkg_name: lower_bound_version} from pyproject.toml."""
    data = tomli.loads(path.read_text())
    project = data.get('project', {})
    reqs: list[str] = list(project.get('dependencies', []))
    for group in data.get('dependency-groups', {}).values():
        reqs.extend(group)
    floors: dict[str, Version] = {}
    for r in reqs:
        req = Requirement(r)
        lb = _lower_bound(req.specifier)
        if lb is None:
            continue
        name = req.name.lower()
        # Take the max: later constraints (e.g. dev-group) only tighten the floor.
        if name not in floors or lb > floors[name]:
            floors[name] = lb
    requires_python = project.get('requires-python')
    if requires_python is not None:
        lb = _lower_bound(SpecifierSet(requires_python))
        if lb is not None:
            floors['python'] = lb
    return floors


def _lower_bound(spec: SpecifierSet) -> Version | None:
    for s in spec:
        if s.operator in ('>=', '==', '~='):
            try:
                return Version(s.version)
            except InvalidVersion:
                return None
    return None


def is_obsolete(op: str, bound: Version, floor: Version) -> bool:
    """Check whether a `version <op> bound` workaround is obsolete.

    It is obsolete when no supported version (>= floor) can satisfy the
    condition.
    """
    if op == '<':
        return floor >= bound
    if op == '<=':
        return floor > bound
    raise ValueError(op)


def scan(root: Path) -> list[tuple[str, str, str]]:
    """Return grep matches `(file, lineno, line)` for WORKAROUND markers."""
    result = subprocess.run(
        ['git', 'grep', '--no-color', '-nI', '-E', r'WORKAROUND\('],  # noqa: S607
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    matches = []
    for line in result.stdout.splitlines():
        f, n, rest = line.split(':', 2)
        if Path(f).resolve() == Path(__file__).resolve():
            continue  # skip this script's own grammar docs
        matches.append((f, n, rest))
    return matches


def uv_floor(root: Path) -> Version:
    """Return the uv version pinned by pre-commit, the only uv we support."""
    text = (root / '.pre-commit-config.yaml').read_text()
    match = UV_REV_RE.search(text)
    if match is None:
        msg = 'cannot find the frozen uv-pre-commit rev in .pre-commit-config.yaml'
        raise ValueError(msg)
    return Version(match[1])


def collect_floors(root: Path) -> dict[str, Version]:
    """Floors for dependencies (pyproject.toml) and `uv` (pre-commit pin)."""
    floors = floors_from_pyproject(root / 'pyproject.toml')
    floors['uv'] = uv_floor(root)
    return floors


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    floors = collect_floors(root)
    stale: list[str] = []
    unknown: list[str] = []
    for file, lineno, text in scan(root):
        candidates = list(CANDIDATE_RE.finditer(text))
        if not candidates:
            unknown.append(
                f'{file}:{lineno}: malformed marker (no closing paren on line): {text.strip()}'
            )
            continue
        for c in candidates:
            inner = c.group(1)
            im = INNER_RE.fullmatch(inner)
            if not im:
                unknown.append(
                    f'{file}:{lineno}: malformed marker {c.group(0)!r}: {text.strip()}'
                )
                continue
            pkg, op, ver = im.group(1).lower(), im.group(2), im.group(3)
            try:
                bound = Version(ver)
            except InvalidVersion:
                unknown.append(
                    f'{file}:{lineno}: bad version in marker: {text.strip()}'
                )
                continue
            floor = floors.get(pkg)
            if floor is None:
                unknown.append(f'{file}:{lineno}: no known floor for {pkg!r}')
                continue
            if is_obsolete(op, bound, floor):
                stale.append(
                    f'{file}:{lineno}: {pkg}{op}{ver} is obsolete (floor={floor}) | {text.strip()}'
                )
    for line in unknown:
        print(f'WARN  {line}', file=sys.stderr)
    for line in stale:
        print(f'STALE {line}', file=sys.stderr)
    return 1 if (stale or unknown) else 0


if __name__ == '__main__':
    sys.exit(main())
