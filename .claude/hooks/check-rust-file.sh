#!/bin/sh
# PostToolUse hook: format and check each edited Rust file.
#
# CI runs `cargo fmt --check` and clippy (see .github/workflows/pytest.yml).
# This hook applies the format here, thus a change needs no second push. clippy
# needs about 1.6 s after a change to a source file, because only the local
# crate compiles again.

set -u

projectdir="${CLAUDE_PROJECT_DIR:-$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)}"
rustdir="$projectdir/rust"

command -v jq >/dev/null 2>&1 || exit 0
command -v cargo >/dev/null 2>&1 || exit 0

# print each path of stdin that names a Rust file of the crate
keep_rust_files() {
    while IFS= read -r filepath; do
        case "$filepath" in
            "$rustdir"/*.rs) [ -f "$filepath" ] && printf '%s\n' "$filepath" ;;
        esac
    done
}

# Claude Code and Codex both run this hook, see get-edited-files.sh
filepaths="$("$(dirname -- "$0")/get-edited-files.sh" | keep_rust_files)"
[ -n "$filepaths" ] || exit 0

cd "$rustdir" || exit 0

# a cold rebuild of the dependencies takes much longer than a rebuild of the crate
if command -v timeout >/dev/null 2>&1; then
    withlimit() { timeout 180 "$@"; }
else
    withlimit() { "$@"; }
fi

# rustfmt reads rust/rustfmt.toml, thus it gives the same result as `cargo fmt`. A path can hold a space, e.g.
# "Mobile Documents", thus the loop splits the list at each line break alone
reformattedfiles=""
set -f
oldifs="$IFS"
IFS='
'
for filepath in $filepaths; do
    before="$(cat -- "$filepath")"
    rustfmt --edition 2024 -- "$filepath" >/dev/null 2>&1
    [ "$before" = "$(cat -- "$filepath")" ] || reformattedfiles="$reformattedfiles $filepath"
done
IFS="$oldifs"
set +f

lints="$(withlimit cargo clippy --all-features --message-format short -- -D warnings 2>&1)"
lintstatus=$?

if [ "$lintstatus" -eq 124 ]; then
    printf 'cargo clippy stopped after 180 s. Run it again in rust/.\n' >&2
    exit 2
fi

if [ "$lintstatus" -eq 0 ] && [ -z "$reformattedfiles" ]; then
    exit 0
fi

{
    if [ -n "$reformattedfiles" ]; then
        printf 'rustfmt changed%s. Read each file again before you edit it.\n\n' "$reformattedfiles"
    fi
    if [ "$lintstatus" -ne 0 ]; then
        printf 'cargo clippy:\n%s\n\n' "$(printf '%s\n' "$lints" | grep -v '^ *Compiling\|^ *Checking\|^ *Finished')"
        printf 'CI runs clippy with -D warnings and the lints of rust/Cargo.toml. Correct each warning.\n'
    fi
} >&2

exit 2
