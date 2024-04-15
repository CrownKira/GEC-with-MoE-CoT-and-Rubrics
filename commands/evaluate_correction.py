import subprocess
import os
import argparse


# NOTE: evaluate on ABCN dev (not test set) since reference m2 for test set is not provided


# python3 commands/parallel_to_m2.py -orig test/ABCN.dev.gold.bea19.first100.orig -cor corrected_output/ABCN.dev.gold.bea19.first100.corrected -out corrected_m2/ABCN.dev.gold.bea19.first100.m2
# python3 commands/compare_m2.py -hyp corrected_m2/ABCN.dev.gold.bea19.first100.m2 -ref reference_m2/ABCN.dev.gold.bea19.first100.m2


# Set up argparse to dynamically extract command line arguments
parser = argparse.ArgumentParser(
    description="Process files for CEFR level evaluation."
)
parser.add_argument(
    "-f",
    "--filename",
    required=True,
    help="CEFR level filename without extension.",
)

# Extract arguments
args = parser.parse_args()

# Extract CEFR_LEVEL_FILENAME from the arguments
CEFR_LEVEL_FILENAME = args.filename


# Define the paths to the input, corrected, and reference files
# CEFR_LEVEL_FILENAME = "ABCN.dev.gold.bea19.first100"
input_file_path = f"./test/{CEFR_LEVEL_FILENAME}.orig"
corrected_file_path = f"./corrected_output/{CEFR_LEVEL_FILENAME}.corrected"
reference_m2_path = (
    f"./reference_m2/{CEFR_LEVEL_FILENAME}.m2"  # true corrections
)
corrected_m2_path = (
    f"./corrected_m2/{CEFR_LEVEL_FILENAME}.m2"  # my corrections
)


# Define the path to the ERRANT scripts
parallel_to_m2_script = "commands/parallel_to_m2.py"
compare_m2_script = "commands/compare_m2.py"


# Check if ERRANT scripts exist
if not os.path.isfile(parallel_to_m2_script):
    raise FileNotFoundError(
        f"ERRANT script not found: {parallel_to_m2_script}"
    )
if not os.path.isfile(compare_m2_script):
    raise FileNotFoundError(f"ERRANT script not found: {compare_m2_script}")


# Step 1: Convert the original and corrected text files to M2 format
subprocess.run(
    [
        "python3",
        parallel_to_m2_script,
        "-orig",
        input_file_path,
        "-cor",
        corrected_file_path,
        "-out",
        corrected_m2_path,
    ],
    check=True,
)

print(f"Converted files to M2 format: {corrected_m2_path}")

# Step 2: Evaluate the system output with the reference M2 file
evaluation_results = subprocess.run(
    [
        "python3",
        compare_m2_script,
        "-hyp",
        corrected_m2_path,
        "-ref",
        reference_m2_path,
    ],
    capture_output=True,
    text=True,
    check=True,
)

# Print the evaluation results
print("Evaluation Results:")
print(evaluation_results.stdout)


# Define the path to the corr_from_m2.py script
corr_from_m2_script = "commands/corr_from_m2.py"

# Define the path for the new output file from the corr_from_m2.py script
corr_from_m2_output = f"./reference_output/{CEFR_LEVEL_FILENAME}.corrected"


# Check if corr_from_m2.py script exists
if not os.path.isfile(corr_from_m2_script):
    raise FileNotFoundError(f"ERRANT script not found: {corr_from_m2_script}")


# TODO: fix this. doesn't apply M2 to the text
# Step 3: Extract corrected text from M2 file
subprocess.run(
    [
        "python3",
        corr_from_m2_script,
        reference_m2_path,
        "-out",
        corr_from_m2_output,
        "-id",
        "0",
    ],
    check=True,
)

print(f"Extracted corrected text to: {corr_from_m2_output}")


# Step 4: Compare the extracted corrected text with the system's corrected text
# def compare_files(file1, file2):
#     with open(file1, "r") as f1, open(file2, "r") as f2:
#         file1_lines = f1.readlines()
#         file2_lines = f2.readlines()

#     differences = []
#     for i, (line1, line2) in enumerate(zip(file1_lines, file2_lines)):
#         if line1 != line2:
#             differences.append((i + 1, line1.strip(), line2.strip()))

#     return differences


# differences = compare_files(corr_from_m2_output, corrected_file_path)
# if differences:
#     print("Differences found between files:")
#     for diff in differences:
#         print(f"Line {diff[0]}: ref={diff[1]} | corr={diff[2]}")
# else:
#     print("No differences found between the files.")

# try:
#     subprocess.run(
#         ["diff", corr_from_m2_output, corrected_file_path],
#         check=True,
#         text=True,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#     )
#     print("No differences found.")
# except subprocess.CalledProcessError as e:
#     # diff exits with a non-zero exit code if there are differences
#     print("Differences found:")
#     print(e.output)


# # Step 4: Instruct the user to manually run diff to compare files
print(
    f"To compare the extracted corrected text with the system's corrected text, run the following command:"
)
print(f"diff {corr_from_m2_output} {corrected_file_path}")
