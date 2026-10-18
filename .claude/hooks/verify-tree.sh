#!/bin/sh
# Stop hook: check the full Python tree after a change.
#
# The PostToolUse hook examines one file. A change to a signature can cause an
# error in a different file, and only a check of the full tree finds it. The
# hook also finds a Python change that a command of Bash made, see the state below.
# AGENTS.md makes both type checkers necessary. prek runs ty, but pyrefly runs
# only in CI, thus this hook is the one place where both run before a push.
#
# Timings on the full tree: ruff 0.5 s, ty 0.5 s, pyrefly 4.1 s.

set -u

projectdir="${CLAUDE_PROJECT_DIR:-$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)}"
# two sessions can work in two checkouts, e.g. a worktree, thus each checkout has its own marker and state
checkoutid="$(printf '%s' "$projectdir" | cksum | cut -d ' ' -f 1)"
marker="${TMPDIR:-/tmp}/artistools-claude-pyedit-$(id -u)-$checkoutid"
statefile="${TMPDIR:-/tmp}/artistools-claude-pystate-$(id -u)-$checkoutid"

hookinput="$(cat)"

# Codex reads a JSON object from the stdout of a Stop hook, and Claude Code accepts one too, thus each stop with no
# error prints an empty object
allow_stop() {
    printf '{}\n'
    exit 0
}

# a second block would repeat without end, thus stop after the first one
if printf '%s' "$hookinput" | jq -e '.stop_hook_active == true' >/dev/null 2>&1; then
    allow_stop
fi

command -v uv >/dev/null 2>&1 || allow_stop
cd "$projectdir" || allow_stop

# The PostToolUse hook writes the marker after an Edit or a Write. A command of Bash can also change a Python file,
# e.g. with sed or a script, and no hook sees that change. Thus the state also holds the commit and a checksum of the
# Python changes, and a different state runs the check. The commits since the last check count only if they change
# a Python file
head="$(git rev-parse HEAD 2>/dev/null)"
# a rename of an untracked file keeps its bytes, thus the names of the untracked files also go in the checksum
changes="$({ git diff HEAD -- '*.py'; git ls-files --others --exclude-standard -- '*.py'; \
    git ls-files --others --exclude-standard -z -- '*.py' | xargs -0 cat 2>/dev/null; } | cksum | cut -d ' ' -f 1)"
state="$head $changes"
laststate="$(cat -- "$statefile" 2>/dev/null)"
lasthead="${laststate%% *}"
printf '%s' "$state" >"$statefile" 2>/dev/null

needscheck=0
[ -f "$marker" ] && needscheck=1
rm -f "$marker"
if [ -n "$laststate" ] && [ "$state" != "$laststate" ]; then
    if [ "${laststate#* }" != "$changes" ]; then
        needscheck=1
    elif [ -n "$(git diff --name-only "$lasthead" "$head" -- '*.py' 2>/dev/null)" ]; then
        needscheck=1
    fi
fi
# the first stop of a checkout has no state. A command of Bash can commit a Python change before that stop, thus
# the first stop runs the check
[ -z "$laststate" ] && needscheck=1
[ "$needscheck" -eq 1 ] || allow_stop

run() { uv run --no-sync -- "$@"; }

lint="$(run ruff check --no-fix 2>&1)"; lintstatus=$?

# a personal ~/.config/ty/ty.toml can add extra search paths that shadow this checkout, e.g. the
# main checkout shadows a worktree. CI reads no personal configuration, thus this check reads none
tycheck="$(XDG_CONFIG_HOME="${TMPDIR:-/tmp}/artistools-hook-no-user-config" uv run --no-sync -- ty check 2>&1)"; tystatus=$?

# the git ignore files hide every path below .claude/worktrees, thus a plain pyrefly check in a
# worktree finds no files and checks nothing. The explicit file list makes pyrefly check the tree
# CI checks an untracked file as well, thus the list holds those files.
# sort -u drops a name that both commands give. The pathspec leaves out the conflict copies that
# iCloud writes, e.g. "core 2.py", because those files are no part of the tree
pyreflycheck="$({ git ls-files -z -- '*.py' ':(exclude)* [0-9].py'; \
    git ls-files -z --others --exclude-standard -- '*.py' ':(exclude)* [0-9].py'; } \
    | sort -z -u | xargs -0 uv run --no-sync -- pyrefly check 2>&1)"; pyreflystatus=$?
pyreflycheck="$(printf '%s\n' "$pyreflycheck" | grep -v '^ *INFO ')"

if [ "$lintstatus" -eq 0 ] && [ "$tystatus" -eq 0 ] && [ "$pyreflystatus" -eq 0 ]; then
    allow_stop
fi

{
    printf 'The full tree check found errors. A change in one file can cause an error in a different file.\n\n'
    [ "$lintstatus" -ne 0 ] && printf 'ruff check:\n%s\n\n' "$lint"
    [ "$tystatus" -ne 0 ] && printf 'ty check:\n%s\n\n' "$tycheck"
    [ "$pyreflystatus" -ne 0 ] && printf 'pyrefly check:\n%s\n\n' "$pyreflycheck"
    printf 'Correct these errors. Both type checkers must give no errors.\n'
} >&2

exit 2
