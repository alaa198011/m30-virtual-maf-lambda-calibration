import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# All configs/*.yaml paths used throughout src/ are relative to the repo
# root, so tests must run with that as the working directory.
os.chdir(ROOT)
