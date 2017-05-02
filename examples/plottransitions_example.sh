#!/usr/bin/env bash

plotartistransitions --no-plot --print-lines -xmin 200 -xmax 30000 -T 2000 -elements Fe > fe_forbidden_lines_2000K.txt
plotartistransitions --no-plot --print-lines -xmin 200 -xmax 30000 -T 8000 -elements Fe > fe_forbidden_lines_8000K.txt
plotartistransitions --no-plot --print-lines -xmin 200 -xmax 30000 -T 2000 -elements Co > co_forbidden_lines_2000K.txt
plotartistransitions --no-plot --print-lines -xmin 200 -xmax 30000 -T 8000 -elements Co > co_forbidden_lines_8000K.txt
