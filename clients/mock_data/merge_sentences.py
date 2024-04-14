import csv


def merge_sentences_to_csv(
    original_file_path: str, corrected_file_path: str, output_csv_path: str
):
    """
    Merges sentences from two files into a CSV file.

    Parameters:
    - original_file_path: Path to the file containing the original sentences.
    - corrected_file_path: Path to the file containing the corrected sentences.
    - output_csv_path: Path to the output CSV file.
    """
    with open(
        original_file_path, "r", encoding="utf-8"
    ) as original_file, open(
        corrected_file_path, "r", encoding="utf-8"
    ) as corrected_file, open(
        output_csv_path, "w", newline="", encoding="utf-8"
    ) as output_csv:

        # Initialize the CSV writer with headers
        csv_writer = csv.writer(output_csv)
        csv_writer.writerow(["Original Sentence", "Corrected Sentence"])

        # Iterate over both files line by line
        for original_line, corrected_line in zip(
            original_file, corrected_file
        ):
            # Trim newline characters and write the pair to the CSV
            csv_writer.writerow(
                [original_line.strip(), corrected_line.strip()]
            )


# Example usage of the function
original_file_path = "./ABCN.dev.gold.bea19.orig"
# corrected_file_path = "./ABCN.dev.gold.bea19.BART-A.corrected"
# output_csv_path = "./ABCN.dev.gold.bea19.BART-A.corrected.csv"
# corrected_file_path = "./ABCN.dev.gold.bea19.BART-B.corrected"
# output_csv_path = "./ABCN.dev.gold.bea19.BART-B.corrected.csv"
# corrected_file_path = "./ABCN.dev.gold.bea19.T5-small-A.corrected"
# output_csv_path = "./ABCN.dev.gold.bea19.T5-small-A.corrected.csv"
corrected_file_path = "./ABCN.dev.gold.bea19.T5-small-B.corrected"
output_csv_path = "./ABCN.dev.gold.bea19.T5-small-B.corrected.csv"


merge_sentences_to_csv(
    original_file_path, corrected_file_path, output_csv_path
)
