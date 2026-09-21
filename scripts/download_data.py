"""Download only missing members of the documented 45-recording subset."""
import argparse
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://physionet.org/files/chbmit/1.0.0"
SUBSET = {
    "chb01": "01 02 03 04 05 10 15 16 18 21 26".split(),
    "chb02": "01 02 05 10 16 16+ 19".split(),
    "chb03": "01 02 03 04 05 06 07 10 34 35 36".split(),
    "chb05": "01 02 03 06 10 13 16 17 22".split(),
    "chb08": "02 03 05 10 11 13 21".split(),
}


def expected_paths():
    for patient, numbers in SUBSET.items():
        yield f"{patient}/{patient}-summary.txt"
        for number in numbers:
            yield f"{patient}/{patient}_{number}.edf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List required files without downloading")
    args = parser.parse_args()
    for relative in expected_paths():
        if args.list:
            print(relative)
            continue
        target = ROOT / "data" / "raw" / relative
        if target.exists():
            if not target.stat().st_size:
                raise ValueError(f"Existing empty source file requires manual inspection: {target}")
            print(f"Preserving existing {relative}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(f"{BASE}/{relative}", timeout=120) as response:
            expected_size = response.headers.get("Content-Length")
            count = 0
            with temporary.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
                    count += len(block)
        if count == 0 or expected_size is not None and count != int(expected_size):
            raise ValueError(f"Incomplete download: {relative}")
        if target.exists():
            raise FileExistsError(target)
        temporary.rename(target)
        print(f"Downloaded {relative}")


if __name__ == "__main__":
    main()
