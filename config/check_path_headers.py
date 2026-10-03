# lsqfitgp/config/check_path_headers.py
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

"""Check the `lsqfitgp/<path>` comment atop each source file matches its location.

Source files open with a comment line `lsqfitgp/<path>` just above the GPL
license header. For files in the package, `<path>` is relative to
`src/lsqfitgp/` (e.g., `lsqfitgp/_GP/_gp.py`); for all other files it is
relative to the repository root (e.g., `lsqfitgp/tests/conftest.py`). The
comment marker varies by language (`#`, `..`, or none inside an HTML comment);
only the path is checked. This catches a header left stale after a file is
moved or renamed.

Files without the lsqfitgp license header are out of scope and skipped. Pass the
files to check as command-line arguments, as pre-commit does.
"""

import re
import sys
from pathlib import Path, PurePosixPath

# Present in every per-file license header but not in the project LICENSE, so it
# scopes the check to files carrying the header.
ANCHOR_RE = re.compile(r'This file is part of lsqfitgp\.')
# The path line, stripped of an optional comment marker, is `lsqfitgp/<path>`.
PATH_RE = re.compile(
    r'^\s*(?:#|\.\.|\{#|/\*|<!-+|\*)?\s*lsqfitgp/(?P<path>.+?)(?:\s*(?:\*/|\#\}|-->))?\s*$'
)
# The package directory, whose files carry paths relative to it.
PACKAGE_DIR = PurePosixPath('src/lsqfitgp')
# Lines scanned for the header, enough to clear an optional shebang or `<!--`.
WINDOW = 10


def expected_path(path: Path) -> str:
    """Return the path expected after `lsqfitgp/` in the header of `path`."""
    posix = PurePosixPath(path.as_posix())
    if posix.is_relative_to(PACKAGE_DIR):
        return posix.relative_to(PACKAGE_DIR).as_posix()
    return posix.as_posix()


def check(path: Path) -> str | None:
    """Return an error message if `path`'s header is wrong, else None."""
    try:
        lines = path.read_text(encoding='utf-8').splitlines()[:WINDOW]
    except (UnicodeDecodeError, FileNotFoundError):
        return None  # binary or deleted file: nothing to check
    anchor = next((i for i, line in enumerate(lines) if ANCHOR_RE.search(line)), None)
    if anchor is None:
        return None  # no license header: out of scope
    for line in lines[:anchor]:
        match = PATH_RE.match(line)
        if match is not None:
            got = match.group('path')
            expected = expected_path(path)
            if got != expected:
                return (
                    f'path header is `lsqfitgp/{got}`, expected `lsqfitgp/{expected}`'
                )
            return None
    return 'license header present but no `lsqfitgp/<path>` line above it'


def main(argv: list[str]) -> int:
    failed = False
    for arg in argv:
        message = check(Path(arg))
        if message is not None:
            print(f'{arg}: {message}', file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
