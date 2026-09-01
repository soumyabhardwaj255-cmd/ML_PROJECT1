"""
During Experiment C, check the number of seizure windows
available for each patient to avoid evaluating F1 on a patient
with very few positive (seizure) windows.
"""

import csv


WINDOW_INDEX_FILE = "data/processed/window_index.csv"
SEIZURE_WINDOWS_FILE = "data/processed/seizure_windows.csv"

PATIENTS = ["chb01", "chb02", "chb03", "chb05", "chb08"]


def extract_seizures():
    """Extract seizure windows from the full window index.

    Reads window_index.csv and saves only the rows labeled as
    seizure windows to seizure_windows.csv.
    """

    with open(
        WINDOW_INDEX_FILE,
        mode="r",
        newline="",
        encoding="utf-8",
    ) as infile:

        reader = csv.reader(infile)

        with open(
            SEIZURE_WINDOWS_FILE,
            mode="w",
            newline="",
            encoding="utf-8",
        ) as outfile:

            writer = csv.writer(outfile)

            headers = ["patient", "file", "start_sec", "end_sec", "label"]
            writer.writerow(headers)

            for row in reader:
                if row[4] == "1":
                    writer.writerow(row)


def count_per_patient():
    """Count the number of seizure windows for each patient.

    Returns:
        dict: A dictionary mapping each patient ID to the number
        of seizure windows associated with that patient.
    """

    seizure_counts = {patient: 0 for patient in PATIENTS}

    with open(
        SEIZURE_WINDOWS_FILE,
        mode="r",
        newline="",
        encoding="utf-8",
    ) as infile:

        reader = csv.reader(infile)

        # Skip the header row.
        next(reader)

        for row in reader:
            patient = row[0]

            if patient in seizure_counts:
                seizure_counts[patient] += 1

    return seizure_counts


if __name__ == "__main__":
    extract_seizures()
    counts = count_per_patient()
    print(counts)