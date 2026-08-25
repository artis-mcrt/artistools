#!/bin/sh
# PreToolUse hook: refuse a direct change to uv.lock.
#
# `uv lock` writes this file. A change by hand passes ruff, pyrefly, and ty, and
# then fails CI, because .github/workflows/pytest.yml sets UV_FROZEN=1.

set -u

filepath="$(jq -r '.tool_input.file_path // empty' 2>/dev/null)"

case "$filepath" in
    */uv.lock | uv.lock) ;;
    *) exit 0 ;;
esac

cat >&2 <<'MSG'
Do not change uv.lock by hand. `uv lock` writes this file, and CI sets
UV_FROZEN=1, thus a change by hand fails the build.

To add or to change a dependency:
  1. Edit the applicable table in pyproject.toml:
     - [project.dependencies] for a run-time dependency;
     - [project.optional-dependencies].extras for a large optional dependency;
     - [dependency-groups].dev for a tool.
  2. Run `uv lock`.
  3. Commit pyproject.toml together with uv.lock.
MSG

exit 2
