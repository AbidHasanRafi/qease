"""Q-EASE pipeline package.

The code lives in `qease/parts/*.py`: ordered, numbered parts of ONE program that `main.py` executes in a single shared namespace
(they are not independent importable modules - they share globals such as CFG, RUNNERS, SELECTORS on purpose, see README.md).
"""
