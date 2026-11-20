# Data reference

This page gives the columns and the units of the main readers of artistools, and the mistakes that are easy to make
with them. The text comes from the `artis-data` skill for a large language model (LLM) agent (see "Use with an LLM
agent" on the home page). Thus the instructions are for the agent, e.g. "give the number of cells that you used". The test
`test_artis_data_skill_examples_run_on_the_test_model` runs each Python example of the skill, thus each example of
this page, on a test model.

Most columns use cgs units. Some columns give their unit at the end of the name, e.g. `_Lsun` or `_kmps`. Each section
below gives the unit of each column, e.g. `f_lambda` is per Å and not per cm.

```{include} ../artistools/data/skills/artis-data/SKILL.md
---
start-after: '<!-- The page docs/data.md shows the text from this line to the heading "Report". -->'
end-before: "## Report"
---
```
