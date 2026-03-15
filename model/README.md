# Model Files

This directory intentionally does not track the large ONNX model files in Git.

Before running the project, make sure these files exist in this folder:

- `encoder.int8.onnx`
- `decoder.int8.onnx`
- `tokens.txt`

Notes:

- `robot_ear.py` will raise an error at startup if any required file is missing.
- The repository ignores `model/*.onnx` so the project can be pushed to GitHub without hitting the normal file size limit.
- If you move the project to another machine, copy the offline-downloaded model files back into this directory before running.
