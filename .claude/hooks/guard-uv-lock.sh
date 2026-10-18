#!/bin/sh
# PreToolUse hook: refuse a direct change to uv.lock.
#
# `uv lock` writes this file. A change by hand passes ruff, pyrefly, and ty, and
# then fails the `uv lock --check` step of .github/workflows/pytest.yml.

set -u

# Claude Code and Codex both run this hook, see get-edited-files.sh
"$(dirname -- "$0")/get-edited-files.sh" | grep -q '/uv\.lock$' || exit 0

cat >&2 <<'MSG'
Do not change uv.lock by hand. `uv lock` writes this file, and CI runs
`uv lock --check`, thus a change by hand fails the build.

To add or to change a dependency:
  1. Edit the applicable table in pyproject.toml:
     - [project.dependencies] for a run-time dependency;
     - [project.optional-dependencies].extras for a large optional dependency;
     - [dependency-groups].dev for a tool.
  2. Run `uv lock`.
  3. Commit pyproject.toml together with uv.lock.
MSG

exit 2
