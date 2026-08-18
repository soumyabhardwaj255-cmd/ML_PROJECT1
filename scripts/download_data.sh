#!/usr/bin/env bash
#
# Downloads a manageable subset of the CHB-MIT Scalp EEG Database:
# 5 patients (chb01, chb02, chb03, chb05, chb08), each with their
# seizure-containing files (positive examples) and a few seizure-free
# files (negative examples), plus each patient's summary file.
#
# Usage:
#   bash download_data.sh
#
# Run this from anywhere — it creates/populates data/raw/ relative to
# the repo root (assumes this script lives in the repo root or scripts/).

set -euo pipefail

BASE_URL="https://physionet.org/files/chbmit/1.0.0"
OUT_DIR="data/raw"

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

# Root-level reference files
echo "Downloading root-level reference files..."
wget -c "$BASE_URL/RECORDS"
wget -c "$BASE_URL/RECORDS-WITH-SEIZURES"
wget -c "$BASE_URL/SUBJECT-INFO"

# ---------------------------------------------------------------------
# Patient file lists
# Format: "patient|seizure_files|nonseizure_files"
# seizure_files = files confirmed to contain a labeled seizure
# nonseizure_files = a few files from the same patient with no seizure,
#                     for negative examples
# ---------------------------------------------------------------------
PATIENTS=(
  "chb01|03 04 15 16 18 21 26|01 02 05 10"
  "chb02|16 19|01 02 05 10"          # note: chb02_16+.edf handled separately below
  "chb03|01 02 03 04 34 35 36|05 06 07 10"
  "chb05|06 13 16 17 22|01 02 03 10"
  "chb08|02 05 11 13 21|01 03 06 10"
)

download_file () {
  local patient=$1
  local fname=$2
  mkdir -p "$patient"
  echo "  -> ${patient}/${fname}"
  wget -c -q --show-progress -P "$patient" "$BASE_URL/${patient}/${fname}" || \
    echo "     WARNING: failed to download ${fname} (may not exist, continuing)"
}

for entry in "${PATIENTS[@]}"; do
  IFS='|' read -r patient seizure_files nonseizure_files <<< "$entry"
  echo ""
  echo "=== $patient ==="

  # Summary file (has exact seizure start/end timestamps per file)
  download_file "$patient" "${patient}-summary.txt"

  # Seizure-containing files: .edf + .edf.seizures annotation
  for f in $seizure_files; do
    download_file "$patient" "${patient}_${f}.edf"
    download_file "$patient" "${patient}_${f}.edf.seizures"
  done

  # Non-seizure files: .edf only (no .seizures annotation exists for these)
  for f in $nonseizure_files; do
    download_file "$patient" "${patient}_${f}.edf"
  done
done

# chb02 has one seizure file with a "+" suffix (chb02_16+.edf) not covered
# by the loop above — grab it explicitly.
echo ""
echo "=== chb02 extra file ==="
download_file "chb02" "chb02_16+.edf.seizures"
download_file "chb02" "chb02_16+.edf"

echo ""
echo "Done. Downloaded data lives under ${OUT_DIR}/"
du -sh . 2>/dev/null || true