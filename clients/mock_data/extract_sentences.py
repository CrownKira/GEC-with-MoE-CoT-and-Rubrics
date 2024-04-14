import csv


def extract_sentences_to_file(
    csv_file_path: str, output_file_path: str, column_name: str = "correct"
):
    """
    Extracts sentences from a specified column in a CSV file and writes them to an output file,
    separated by newlines.

    Parameters:
    - csv_file_path: Path to the input CSV file.
    - output_file_path: Path to the output file where sentences will be written.
    - column_name: The name of the column to extract sentences from. Defaults to 'correct'.
    """
    sentences = []

    # Read the specified column from the CSV file
    with open(csv_file_path, "r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if column_name in row:
                sentences.append(row[column_name].strip())

    # Write the extracted sentences to the output file, separated by newlines
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(sentences))


# Example usage
csv_file_path = "ABCN.dev.gold.bea19.T5-small-B.corrected.csv"
output_file_path = "ABCN.dev.gold.bea19.T5-small-B.corrected"


extract_sentences_to_file(csv_file_path, output_file_path)
