# Working on this repo

## Visual testing is done by the maintainer only

Do not test the add-on visually: no windowed Blender sessions, no screenshots of
Blender (MCP screenshot tools included), no looking at rendered UI to judge a change.
Verify with the unit tests (`python -m unittest discover -s tests`) and the headless
Blender runs: `tests/run_in_blender.py --package`, and with
`--script tests/reference/blender_reference.py` for the reference curves. When a
change needs a visual check (panel layout, drawing, the Node Editor), list what to
look at and leave it to the maintainer.
