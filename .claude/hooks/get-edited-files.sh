#!/bin/sh
# Print the path of each file that the edit of a hook input names, one path on each line.
#
# Claude Code gives the path in tool_input.file_path. Codex edits with apply_patch, and it gives the text of the patch
# in tool_input.command. Each "*** Add File:", "*** Update File:", and "*** Move to:" line of a patch names a file. A
# relative path starts at the cwd of the hook input.

set -u

command -v jq >/dev/null 2>&1 || exit 0

hookinput="$(cat)"
cwd="$(printf '%s' "$hookinput" | jq -r '.cwd // empty' 2>/dev/null)"

{
    printf '%s' "$hookinput" | jq -r '.tool_input.file_path // empty' 2>/dev/null
    printf '%s' "$hookinput" | jq -r '.tool_input.command // empty' 2>/dev/null \
        | sed -n -E 's/^\*\*\* (Add File|Update File|Move to): //p'
} | while IFS= read -r filepath; do
    [ -n "$filepath" ] || continue
    case "$filepath" in
        /*) printf '%s\n' "$filepath" ;;
        *) printf '%s\n' "${cwd:-$PWD}/$filepath" ;;
    esac
done
