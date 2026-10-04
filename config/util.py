# lsqfitgp/config/util.py
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

"""Release helpers driven by the topmost section of the changelog.

The changelog section headers look like ``## 0.21.2. Title (2026-10-03)``: the
version (two or more numeric components, matching the ``v<version>`` git tag),
a period, the release title, and the release date.
"""

import argparse
import datetime
import re
import subprocess
from pathlib import Path

CHANGELOG_PATH = Path('docs/development/changelog.md')


def parse_datetime(s: str) -> datetime.datetime:
    """Parse an RFC 3339 timestamp, requiring an explicit timezone offset."""
    dt = datetime.datetime.fromisoformat(s.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        msg = f'timestamp {s!r} must include a timezone offset'
        raise argparse.ArgumentTypeError(msg)
    return dt


def get_version() -> str:
    """Read the release version from the topmost changelog section."""
    for line in CHANGELOG_PATH.read_text().splitlines():
        if line.startswith('## '):
            version, _title, _date = _parse_changelog_header(line)
            return version
    msg = f'No release header found in {CHANGELOG_PATH}'
    raise ValueError(msg)


def _parse_changelog_header(line: str) -> tuple[str, str, str]:
    """Parse a ``## VERSION. TITLE (YYYY-MM-DD)`` line, error if malformed."""
    m = re.fullmatch(r'## (\d+(?:\.\d+)+)\. (.+) \((\d{4}-\d{2}-\d{2})\)', line)
    if m is None:
        msg = f'Cannot parse changelog header: {line!r}'
        raise ValueError(msg)
    return m[1], m[2], m[3]


def _read_changelog_section(today: datetime.date) -> tuple[str, str, str, str]:
    """Read and validate the topmost changelog section.

    Returns ``(version, title, date, body)``. Raises ValueError if the
    changelog cannot be parsed or the date is not ``today``.
    """
    lines = CHANGELOG_PATH.read_text().splitlines()
    headers = [i for i, line in enumerate(lines) if line.startswith('## ')]
    if len(headers) < 2:
        msg = f'Expected at least 2 release headers in {CHANGELOG_PATH}, found {len(headers)}'
        raise ValueError(msg)
    first, second = headers[0], headers[1]
    version, title, date = _parse_changelog_header(lines[first])
    _parse_changelog_header(lines[second])  # validate boundary header
    if date != today.isoformat():
        msg = f'Changelog date {date} does not match today {today.isoformat()}'
        raise ValueError(msg)
    body = '\n'.join(lines[first + 1 : second]).strip('\n')
    return version, title, date, body


def check_changelog(today: datetime.date) -> None:
    """Validate the topmost changelog section is parseable and dated ``today``."""
    _read_changelog_section(today)


def gh_release(today: datetime.date) -> None:
    """Create a draft GitHub release from the topmost changelog section."""
    version, title, _date, body = _read_changelog_section(today)
    subprocess.run(  # noqa: S603
        [  # noqa: S607
            'gh',
            'release',
            'create',
            f'v{version}',
            '--draft',
            '--verify-tag',
            '--title',
            title,
            '--notes-file',
            '-',
        ],
        input=body,
        text=True,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'command', choices=('get_version', 'check_changelog', 'gh_release')
    )
    parser.add_argument(
        '--today',
        type=parse_datetime,
        help='RFC 3339 timestamp (with timezone); reference instant for the changelog date check.',
    )
    args = parser.parse_args()

    if args.command == 'get_version':
        print(get_version())
    else:
        if args.today is None:
            parser.error(f'{args.command} requires --today')
        today = args.today.astimezone(datetime.timezone.utc).date()
        if args.command == 'check_changelog':
            check_changelog(today)
        elif args.command == 'gh_release':
            gh_release(today)


if __name__ == '__main__':
    main()
