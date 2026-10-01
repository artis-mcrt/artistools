"""Functions that each window of the --interactive option shares, e.g. plotspectra and plotestimators.

A viewer makes the command of the plot from its controls, then parses the command and draws the plot. The
modules here make and change the command tokens, and they build the parts of the Qt window that do not
depend on the command. Each function that uses Qt imports PySide6 when it runs, because the CLI must start
quickly and PySide6 is an optional dependency.
"""
