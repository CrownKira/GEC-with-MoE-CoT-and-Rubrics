import csv
from typing import List, Tuple
import random
import sys

FILENAME = "DataSet_Misinfo_first100"
FILE_PATH_TRUE = f"./datasets/DataSet_Misinfo_TRUE_first100.csv"
FILE_PATH_FAKE = f"./datasets/DataSet_Misinfo_FAKE_first100.csv"
MERGED_FILE_PATH = f"./test/{FILENAME}.orig"
TRUTH_FILE_PATH = f"./reference_output/{FILENAME}.corrected"


# Increase the maximum CSV field size allowed
csv.field_size_limit(sys.maxsize)


def read_csv_file(file_path: str) -> List[str]:
    """Read a CSV file and return a list of texts."""
    texts = []
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            texts.append(row["text"])
    return texts


def write_to_files(
    merged_texts: List[str],
    truth_values: List[str],
    merged_file_path: str,
    truth_file_path: str,
) -> None:
    """Write merged texts and their corresponding truth values to files."""
    with open(merged_file_path, mode="w", encoding="utf-8") as merged_file:
        for text in merged_texts:
            merged_file.write(text + "\n")

    with open(truth_file_path, mode="w", encoding="utf-8") as truth_file:
        for value in truth_values:
            truth_file.write(value + "\n")


def merge_and_shuffle(
    file_path_true: str, file_path_fake: str
) -> Tuple[List[str], List[str]]:
    """Merge and shuffle texts from two files, returning the shuffled texts and their truth values."""
    true_texts = read_csv_file(file_path_true)
    fake_texts = read_csv_file(file_path_fake)

    # Create a combined list of tuples where each tuple is (text, truth_value)
    combined = [(text, "true") for text in true_texts] + [
        (text, "false") for text in fake_texts
    ]

    random.shuffle(combined)

    # Unzip the shuffled combined list into two lists
    merged_texts, truth_values = zip(*combined)

    return list(merged_texts), list(truth_values)


def main():

    merged_texts, truth_values = merge_and_shuffle(FILE_PATH_TRUE, FILE_PATH_FAKE)
    write_to_files(merged_texts, truth_values, MERGED_FILE_PATH, TRUTH_FILE_PATH)

    print(f"Merged data written to {MERGED_FILE_PATH}")
    print(f"Corresponding truth values written to {TRUTH_FILE_PATH}")


if __name__ == "__main__":
    main()
