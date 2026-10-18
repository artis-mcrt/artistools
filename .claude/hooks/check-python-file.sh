#!/bin/sh
# PostToolUse hook: format and check each edited Python file.
#
# pyrefly runs only in CI (see .github/workflows/pytest.yml), thus a type error
# stays hidden until a push. This hook gives the same result in under a second.
# The hook examines the edited files. Run `pyrefly check` on the full tree before
# a commit, because a change in one file can cause an error in a different file.
# Claude Code and Codex both run this hook, see get-edited-files.sh.

set -u

projectdir="${CLAUDE_PROJECT_DIR:-$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)}"

command -v uv >/dev/null 2>&1 || exit 0
command -v jq >/dev/null 2>&1 || exit 0

# print each path of stdin that names a Python file of the repository. A file outside it uses different settings
keep_python_files() {
    while IFS= read -r filepath; do
        case "$filepath" in
            "$projectdir"/*.py) [ -f "$filepath" ] && printf '%s\n' "$filepath" ;;
        esac
    done
}

# the hook receives the tool call as JSON on stdin
filepaths="$("$(dirname -- "$0")/get-edited-files.sh" | keep_python_files)"
[ -n "$filepaths" ] || exit 0

cd "$projectdir" || exit 0

# the Stop hook reads this marker, thus the full tree check runs after a change. Two sessions can work in two
# checkouts, e.g. a worktree, thus the name of the marker holds a checksum of the folder of the checkout
checkoutid="$(printf '%s' "$projectdir" | cksum | cut -d ' ' -f 1)"
touch "${TMPDIR:-/tmp}/artistools-claude-pyedit-$(id -u)-$checkoutid" 2>/dev/null

# --no-sync keeps the hook from a reinstall of the environment at each edit
run() { uv run --no-sync -- "$@"; }

# check one file, write each problem to stderr, and return 1 if the file has a problem
check_file() {
    filepath="$1"
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
        return 0
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
            printf 'Correct these errors. Add a suppression only if you cannot correct the problem.\n\n'
        fi
    } >&2
    return 1
}

# a patch of Codex can change several files, thus each file gets its own check
failed=0
# a path can hold a space, e.g. "Mobile Documents", thus the loop splits the list at each line break alone
set -f
oldifs="$IFS"
IFS='
'
for filepath in $filepaths; do
    check_file "$filepath" || failed=1
done
IFS="$oldifs"
set +f

[ "$failed" -eq 0 ] || exit 2
exit 0
