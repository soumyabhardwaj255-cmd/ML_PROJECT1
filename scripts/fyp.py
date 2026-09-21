"""Run from any working directory using the copied project's environment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eeg_seizure.cli import main

if __name__ == "__main__":
    main()
