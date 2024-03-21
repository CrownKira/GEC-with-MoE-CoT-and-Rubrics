import os

# Define the path to the M2 file and the output file
m2_file_path = "reference_m2/ABCN.dev.gold.bea19.m2"
output_file_path = "test/ABCN.dev.gold.bea19.m2.orig"

# Function to extract and tokenize sentences from an M2 file
def extract_sentences_from_m2(m2_file, output_file):
    with open(m2_file, "r") as m2f, open(output_file, "w") as outf:
        for line in m2f:
            if line.startswith("S "):  # Check if the line is a source sentence
                # Extract the sentence without the 'S ' prefix
                sentence = line[2:].strip()
                # Write the tokenized sentence to the output file
                outf.write(sentence + "\n")


# Run the function with the specified file paths
if __name__ == "__main__":
    extract_sentences_from_m2(m2_file_path, output_file_path)
    print(f"Tokenized sentences have been written to {output_file_path}")
