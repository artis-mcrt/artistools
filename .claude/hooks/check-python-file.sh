#!/bin/sh
# PostToolUse hook: format and check one edited Python file.
#
# pyrefly runs only in CI (see .github/workflows/pytest.yml), thus a type error
# stays hidden until a push. This hook gives the same result in under a second.
# The hook examines one file. Run `pyrefly check` on the full tree before a
# commit, because a change in one file can cause an error in a different file.

set -u

projectdir="${CLAUDE_PROJECT_DIR:-$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)}"

# the hook receives the tool call as JSON on stdin
filepath="$(jq -r '.tool_input.file_path // empty' 2>/dev/null)"

[ -n "$filepath" ] || exit 0
case "$filepath" in
    *.py) ;;
    *) exit 0 ;;
esac
case "$filepath" in
    "$projectdir"/*) ;;
    *) exit 0 ;;  # a file outside the repository uses different settings
esac
[ -f "$filepath" ] || exit 0
command -v uv >/dev/null 2>&1 || exit 0
command -v jq >/dev/null 2>&1 || exit 0

cd "$projectdir" || exit 0

# the Stop hook reads this marker, thus the full tree check runs only after a change
touch "${TMPDIR:-/tmp}/artistools-claude-pyedit-$(id -u)" 2>/dev/null

# --no-sync keeps the hook from a reinstall of the environment at each edit
run() { uv run --no-sync -- "$@"; }

before="$(cat -- "$filepath")"
run ruff format --quiet -- "$filepath" >/dev/null 2>&1
reformatted=0
[ "$before" = "$(cat -- "$filepath")" ] || reformatted=1

lint="$(run ruff check --no-fix -- "$filepath" 2>&1)"
lintstatus=$?

types="$(run pyrefly check -- "$filepath" 2>&1)"
typestatus=$?
types="$(printf '%s\n' "$types" | grep -v '^ *INFO ' )"

if [ "$lintstatus" -eq 0 ] && [ "$typestatus" -eq 0 ] && [ "$reformatted" -eq 0 ]; then
    exit 0
fi

{
    if [ "$reformatted" -eq 1 ]; then
        printf 'ruff format changed %s. Read the file again before you edit it.\n\n' "$filepath"
    fi
    if [ "$lintstatus" -ne 0 ]; then
        printf 'ruff check:\n%s\n\n' "$lint"
    fi
    if [ "$typestatus" -ne 0 ]; then
        printf 'pyrefly check:\n%s\n\n' "$types"
    fi
    if [ "$lintstatus" -ne 0 ] || [ "$typestatus" -ne 0 ]; then
        printf 'Correct these errors. Add a suppression only if you cannot correct the problem.\n'
    fi
} >&2

exit 2
