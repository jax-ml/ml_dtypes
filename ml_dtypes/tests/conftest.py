import sys
from pathlib import Path

# Add ml_dtypes/tests folder to discover multi_thread_utils.py module
sys.path.insert(0, str(Path(__file__).absolute().parent))
