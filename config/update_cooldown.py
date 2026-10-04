# lsqfitgp/config/update_cooldown.py
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

"""Pin the dependency cooldown in pyproject.toml to an absolute instant.

uv's `[tool.uv] exclude-newer` accepts a relative span (e.g. `1 week`), but a
span re-resolves against wall-clock time, so `make release` and the uv-lock
pre-commit hook would compute different cutoffs and churn the lockfile. Pin it
to `today - cooldown_days` as an RFC 3339 UTC instant instead: an absolute
value both agree on, anchored to the UTC day so same-day runs are reproducible.
"""

import argparse
import datetime
import re
import sys
from pathlib import Path


def parse_datetime(s: str) -> datetime.datetime:
    """Parse an RFC 3339 timestamp, requiring an explicit timezone offset."""
    dt = datetime.datetime.fromisoformat(s.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        msg = f'timestamp {s!r} must include a timezone offset'
        raise argparse.ArgumentTypeError(msg)
    return dt


def format_utc(dt: datetime.datetime) -> str:
    """Format an instant as an RFC 3339 UTC timestamp with a trailing `Z`."""
    return dt.astimezone(datetime.timezone.utc).isoformat().replace('+00:00', 'Z')


def pin_exclude_newer(pyproject_path: Path, cutoff: datetime.datetime) -> bool:
    """Rewrite the `exclude-newer` value in pyproject.toml; True if changed."""
    text = pyproject_path.read_text()
    pattern = re.compile(r'^exclude-newer = .*$', re.MULTILINE)
    new_text, n = pattern.subn(f"exclude-newer = '{format_utc(cutoff)}'", text)
    if n != 1:
        msg = f'expected one exclude-newer line in {pyproject_path}, found {n}'
        raise RuntimeError(msg)
    if new_text == text:
        return False
    pyproject_path.write_text(new_text)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--today',
        required=True,
        type=parse_datetime,
        help='RFC 3339 timestamp (with timezone); reference instant for the policy.',
    )
    parser.add_argument(
        '--cooldown-days',
        required=True,
        type=int,
        help='Days to hold back from today (the dependency cooldown).',
    )
    parser.add_argument('--pyproject', type=Path, default=Path('pyproject.toml'))
    args = parser.parse_args()

    cutoff = args.today - datetime.timedelta(days=args.cooldown_days)
    changed = pin_exclude_newer(args.pyproject, cutoff)
    print(
        f'exclude-newer -> {format_utc(cutoff)} ({"updated" if changed else "unchanged"})'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
