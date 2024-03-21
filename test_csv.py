# import csv


# def check_csv_for_line_breaks(csv_file_path):
#     with open(csv_file_path, newline="", encoding="utf-8") as csvfile:
#         reader = csv.reader(csvfile)
#         row_number = 0
#         for row in reader:
#             row_number += 1
#             for field in row:
#                 if "\n" in field:
#                     print(
#                         f"Line break found in row {row_number}, field: {field}"
#                     )
#                     break


# if __name__ == "__main__":
#     csv_file_path = "corrected_output/ABCN.dev.gold.bea19.first100.corrected.csv"  # Change this to the path of your CSV file
#     check_csv_for_line_breaks(csv_file_path)


# import csv


# def check_csv_for_line_breaks(csv_file_path):
#     with open(csv_file_path, newline="", encoding="utf-8") as csvfile:
#         reader = csv.DictReader(csvfile)
#         row_number = 0
#         for row in reader:
#             row_number += 1
#             for field_name, field_value in row.items():
#                 if "\n" in field_value:
#                     print(
#                         f"Line break found in row {row_number}, field '{field_name}': {field_value!r}"
#                     )
#                     # Using !r to show the string representation including special characters


# if __name__ == "__main__":
#     csv_file_path = "corrected_output/ABCN.dev.gold.bea19.first100.corrected.csv"  # Change this to the path of your CSV file
#     check_csv_for_line_breaks(csv_file_path)


import csv
from operator import itemgetter


def check_csv_for_line_breaks(csv_file_path):
    rows = []

    # Read the CSV and collect rows
    with open(csv_file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)

    # Sort the rows by Batch Number (assuming it's convertable to integer)
    sorted_rows = sorted(rows, key=lambda row: int(row["Batch Number"]))

    # Check for line breaks in the Corrected Text of each sorted row
    for row in sorted_rows:
        batch_number = row["Batch Number"]
        corrected_text = row.get(
            "Corrected Text", ""
        )  # Using .get() to avoid KeyError if the field is missing
        # if "\n" in corrected_text:
        #     print(
        #         f"Line break found in Batch Number {batch_number}, Corrected Text: {corrected_text!r}"
        #     )

        print("corrected_lines:", row["Corrected Text"].split("\n"))
        print("line number:", len(row["Corrected Text"].split("\n")))


if __name__ == "__main__":
    csv_file_path = "corrected_output/ABCN.dev.gold.bea19.first100.corrected.csv"  # Change this to the path of your CSV file
    check_csv_for_line_breaks(csv_file_path)
