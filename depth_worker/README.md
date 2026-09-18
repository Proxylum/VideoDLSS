# depth_worker (stage 2)

Python (PyTorch + CUDA) worker running ICDepth — the "high quality" depth mode. This is the only
Python component in the runtime; it exchanges frames and passes with the core through shared memory /
pass files. Comes with its own `venv` and `requirements.txt` in stage 2.
