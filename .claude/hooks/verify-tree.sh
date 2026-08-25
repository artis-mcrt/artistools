#!/bin/sh
# Stop hook: check the full Python tree after a change.
#
# The PostToolUse hook examines one file. A change to a signature can cause an
# error in a different file, and only a check of the full tree finds it.
# AGENTS.md makes both type checkers necessary. prek runs ty, but pyrefly runs
# only in CI, thus this hook is the one place where both run before a push.
#
# Timings on the full tree: ruff 0.5 s, ty 0.5 s, pyrefly 4.1 s.

set -u

projectdir="${CLAUDE_PROJECT_DIR:-$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)}"
marker="${TMPDIR:-/tmp}/artistools-claude-pyedit-$(id -u)"

hookinput="$(cat)"

# a second block would repeat without end, thus stop after the first one
if printf '%s' "$hookinput" | jq -e '.stop_hook_active == true' >/dev/null 2>&1; then
    exit 0
fi

# the PostToolUse hook writes the marker, thus no Python change means no check
[ -f "$marker" ] || exit 0
rm -f "$marker"

command -v uv >/dev/null 2>&1 || exit 0
cd "$projectdir" || exit 0

run() { uv run --no-sync -- "$@"; }

lint="$(run ruff check --no-fix 2>&1)"; lintstatus=$?
tycheck="$(run ty check 2>&1)"; tystatus=$?
pyreflycheck="$(run pyrefly check 2>&1)"; pyreflystatus=$?
pyreflycheck="$(printf '%s\n' "$pyreflycheck" | grep -v '^ *INFO ')"

if [ "$lintstatus" -eq 0 ] && [ "$tystatus" -eq 0 ] && [ "$pyreflystatus" -eq 0 ]; then
    exit 0
fi

{
    printf 'The full tree check found errors. A change in one file can cause an error in a different file.\n\n'
    [ "$lintstatus" -ne 0 ] && printf 'ruff check:\n%s\n\n' "$lint"
    [ "$tystatus" -ne 0 ] && printf 'ty check:\n%s\n\n' "$tycheck"
    [ "$pyreflystatus" -ne 0 ] && printf 'pyrefly check:\n%s\n\n' "$pyreflycheck"
    printf 'Correct these errors. Both type checkers must give no errors.\n'
} >&2

exit 2
