#!/bin/sh
# PostToolUse hook: format and check one edited Rust file.
#
# CI runs clippy but never `cargo fmt` (see .github/workflows/pytest.yml), thus
# the format of the Rust code has no gate at all. clippy needs about 1.6 s after
# a change to a source file, because only the local crate compiles again.

set -u

projectdir="${CLAUDE_PROJECT_DIR:-$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)}"
rustdir="$projectdir/rust"

filepath="$(jq -r '.tool_input.file_path // empty' 2>/dev/null)"

[ -n "$filepath" ] || exit 0
case "$filepath" in
    "$rustdir"/*.rs) ;;
    *) exit 0 ;;
esac
[ -f "$filepath" ] || exit 0
command -v jq >/dev/null 2>&1 || exit 0
command -v cargo >/dev/null 2>&1 || exit 0

cd "$rustdir" || exit 0

# a cold rebuild of the dependencies takes much longer than a rebuild of the crate
if command -v timeout >/dev/null 2>&1; then
    withlimit() { timeout 180 "$@"; }
else
    withlimit() { "$@"; }
fi

# rustfmt reads rust/rustfmt.toml, thus it gives the same result as `cargo fmt`
before="$(cat -- "$filepath")"
rustfmt --edition 2024 -- "$filepath" >/dev/null 2>&1
reformatted=0
[ "$before" = "$(cat -- "$filepath")" ] || reformatted=1

lints="$(withlimit cargo clippy --all-features --message-format short -- -D warnings -D clippy::pedantic 2>&1)"
lintstatus=$?

if [ "$lintstatus" -eq 124 ]; then
    printf 'cargo clippy stopped after 180 s. Run it again in rust/.\n' >&2
    exit 2
fi

if [ "$lintstatus" -eq 0 ] && [ "$reformatted" -eq 0 ]; then
    exit 0
fi

{
    if [ "$reformatted" -eq 1 ]; then
        printf 'rustfmt changed %s. Read the file again before you edit it.\n\n' "$filepath"
    fi
    if [ "$lintstatus" -ne 0 ]; then
        printf 'cargo clippy:\n%s\n\n' "$(printf '%s\n' "$lints" | grep -v '^ *Compiling\|^ *Checking\|^ *Finished')"
        printf 'CI runs clippy with -D warnings -D clippy::pedantic. Correct each warning.\n'
    fi
} >&2

exit 2
