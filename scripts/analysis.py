"""Separate bounded PSD/SR/CR analysis entry point; never starts model training."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from eeg_seizure.analysis.runner import main

if __name__ == "__main__":
    main()
