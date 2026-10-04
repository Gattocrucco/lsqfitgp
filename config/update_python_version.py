# lsqfitgp/config/update_python_version.py
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

"""Update the supported Python version range.

Policy:
- A new Python minor is adopted each year on BUMP_PYTHON_VERSION_DATE (MM-DD).
- We support the most recent `num_supported` minor releases.
- Anchor: in ANCHOR_YEAR (2026), before the bump date of that year, the latest
  supported Python minor is ANCHOR_LATEST_MINOR (14). One bump = +1 minor.
- Non-regression: neither the requires-python floor nor the pinned latest
  Python version ever moves backward.

Outputs:
- Rewrites `requires-python = ">=3.<oldest>"` in pyproject.toml.
- Rewrites .python-version to `3.<latest>`.
"""

import argparse
import datetime
import re
import sys
from pathlib import Path

import tomli
from packaging.specifiers import SpecifierSet
from packaging.version import Version

ANCHOR_YEAR = 2026
ANCHOR_LATEST_MINOR = 14  # latest Python = 3.14 in 2026 before the bump date


def parse_bump_date(s: str) -> tuple[int, int]:
    m = re.fullmatch(r'(\d{1,2})-(\d{1,2})', s)
    if not m:
        msg = f'invalid --bump-date {s!r}, expected MM-DD'
        raise argparse.ArgumentTypeError(msg)
    month, day = int(m.group(1)), int(m.group(2))
    datetime.date(2000, month, day)  # validate
    return month, day


def parse_datetime(s: str) -> datetime.datetime:
    """Parse an RFC 3339 timestamp, requiring an explicit timezone offset."""
    dt = datetime.datetime.fromisoformat(s.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        msg = f'timestamp {s!r} must include a timezone offset'
        raise argparse.ArgumentTypeError(msg)
    return dt


def compute_latest_minor(
    today: datetime.datetime,
    bump: tuple[int, int],
    anchor_year: int,
    anchor_latest: int,
) -> int:
    bumps = today.year - anchor_year
    if (today.month, today.day) >= bump:
        bumps += 1
    return anchor_latest + bumps


def read_requires_python_floor(pyproject: dict) -> Version | None:
    spec_str = pyproject.get('project', {}).get('requires-python')
    if not spec_str:
        return None
    spec = SpecifierSet(spec_str)
    for s in spec:
        if s.operator in ('>=', '==', '~='):
            return Version(s.version)
    return None


def read_pinned_python(path: Path) -> Version | None:
    if not path.exists():
        return None
    content = path.read_text().strip()
    if not content:
        return None
    return Version(content.splitlines()[0].strip())


def update_requires_python(path: Path, new_floor: Version) -> bool:
    text = path.read_text()
    pattern = re.compile(r'^(requires-python\s*=\s*)"[^"]*"', re.MULTILINE)
    new_line = f'requires-python = ">=3.{new_floor.minor}"'

    def repl(_m: re.Match[str]) -> str:
        return new_line

    new_text, n = pattern.subn(repl, text, count=1)
    if n == 0:
        msg = 'requires-python assignment not found in pyproject.toml'
        raise RuntimeError(msg)
    if new_text == text:
        return False
    path.write_text(new_text)
    return True


def update_python_version_file(path: Path, new_latest: Version) -> bool:
    new_content = f'3.{new_latest.minor}\n'
    old_content = path.read_text() if path.exists() else ''
    if old_content == new_content:
        return False
    path.write_text(new_content)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--bump-date',
        required=True,
        type=parse_bump_date,
        help='MM-DD; day of year on which a new Python minor is adopted.',
    )
    parser.add_argument(
        '--num-supported',
        required=True,
        type=int,
        help='How many consecutive Python minors to support.',
    )
    parser.add_argument('--pyproject', type=Path, default=Path('pyproject.toml'))
    parser.add_argument(
        '--python-version-file', type=Path, default=Path('.python-version')
    )
    parser.add_argument(
        '--today',
        required=True,
        type=parse_datetime,
        help='RFC 3339 timestamp (with timezone); reference instant for the policy.',
    )
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    today = args.today
    latest_minor = compute_latest_minor(
        today, args.bump_date, ANCHOR_YEAR, ANCHOR_LATEST_MINOR
    )
    oldest_minor = latest_minor - (args.num_supported - 1)

    with args.pyproject.open('rb') as f:
        pyproject = tomli.load(f)
    current_floor = read_requires_python_floor(pyproject)
    current_pinned = read_pinned_python(args.python_version_file)

    computed_floor = Version(f'3.{oldest_minor}')
    computed_latest = Version(f'3.{latest_minor}')
    new_floor = max(computed_floor, current_floor) if current_floor else computed_floor
    new_latest = (
        max(computed_latest, current_pinned) if current_pinned else computed_latest
    )

    print(
        f'Python policy: today={today.isoformat()} -> '
        f'computed range 3.{oldest_minor}..3.{latest_minor}'
    )
    print(
        f'requires-python floor: {current_floor} -> {new_floor} '
        f'({"unchanged" if current_floor == new_floor else "advanced"})'
    )
    print(
        f'.python-version: {current_pinned} -> {new_latest} '
        f'({"unchanged" if current_pinned == new_latest else "advanced"})'
    )

    if args.dry_run:
        return 0

    py_changed = update_requires_python(args.pyproject, new_floor)
    pv_changed = update_python_version_file(args.python_version_file, new_latest)
    print(f'pyproject.toml requires-python: {"updated" if py_changed else "unchanged"}')
    print(f'.python-version: {"updated" if pv_changed else "unchanged"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
