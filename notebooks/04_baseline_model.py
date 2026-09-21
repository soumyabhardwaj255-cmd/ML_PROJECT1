"""Compatibility entry point. Uses the canonical corrected pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eeg_seizure.cli import main

if __name__ == "__main__":
    main(['evaluate', '--kind', 'ml', '--models', 'random_forest'] + sys.argv[1:])
