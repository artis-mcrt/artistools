#!/bin/sh
# SessionStart hook: report the iCloud Drive conflict copies.
#
# This repository is below ~/Library/Mobile Documents. iCloud Drive keeps a
# second copy of a file that it cannot merge, and it adds " 2" or " 3" to the
# name. Such a copy in tests/data can confuse a search for a data file.
# git ignores these copies, thus no other check reports them.

set -u

projectdir="${CLAUDE_PROJECT_DIR:-$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)}"

case "$projectdir" in
    *"/Mobile Documents/"*) ;;
    *) exit 0 ;;  # the repository is outside iCloud Drive
esac

cd "$projectdir" || exit 0

# the search covers the source and the test data only, because rust/target and
# .venv hold many files and no file that a person edits. A cache directory holds
# a copy that a tool writes again, thus the search leaves it out.
copies="$(find artistools tests \( -name '__pycache__' -o -name '.*_cache' \) -prune \
    -o -type f -name '* [0-9].*' -print 2>/dev/null)"
count="$(printf '%s\n' "$copies" | grep -c . )"

[ "$count" -gt 0 ] || exit 0

printf 'iCloud Drive conflict copies: %s files below artistools/ and tests/.\n' "$count"
printf 'Examples:\n'
printf '%s\n' "$copies" | head -3 | sed 's/^/  /'
printf 'To delete them: find artistools tests \\( -name "__pycache__" -o -name ".*_cache" \\) -prune -o -type f -name "* [0-9].*" -print -delete\n'
printf 'Ask the user first, because the pattern can match a real file name.\n'

exit 0
