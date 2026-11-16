import os
import sys
import tempfile
from pathlib import Path

# Keep test data out of the real data folder; must be set before engine is imported.
os.environ["SCRIPTOR_DATA"] = tempfile.mkdtemp(prefix="scriptor-test-")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
