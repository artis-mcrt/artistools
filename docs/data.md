# Data reference

This page gives the columns and the units of the main readers of artistools, and the mistakes that are easy to make
with them. The text comes from the `artis-data` skill for a large language model (LLM) agent (see "Use with an LLM
agent" on the home page). Thus the instructions are for the agent, e.g. "give the number of cells that you used". A
test runs each Python example on the test model.

All values are in cgs units, unless the name of a column gives a different unit, e.g. `_Lsun` or `_kmps`.

```{include} ../artistools/data/skills/artis-data/SKILL.md
---
start-after: '<!-- The page docs/data.md shows the text from this line to the heading "Report". -->'
end-before: "## Report"
---
```
