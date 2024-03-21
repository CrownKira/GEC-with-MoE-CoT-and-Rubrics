import os
import sys
import json
import openai
import asyncio
import logging
import aiofiles
from dotenv import load_dotenv
from csv import DictReader, DictWriter
import atexit
from typing import Any
import spacy
import logging
import datetime


# python3 main.py
# python3 commands/evaluate_correction.py


# python3 commands/corr_from_m2.py reference_m2/ABCN.dev.gold.bea19.first100.m2 -out reference_output/ABCN.dev.gold.bea19.first100.corrected -id 0
# then compare corrected_output with reference_output


# Load the spaCy model outside of the asynchronous function to avoid reloading it multiple times
nlp = spacy.load("en_core_web_sm")


# Load environment variables from .env file
load_dotenv()


# Configuration variables

# ANSI escape codes for colors
RED = "\033[1;31m"
GREEN = "\033[1;32m"
YELLOW = "\033[93m"
BLUE = "\033[1;34m"
RESET = "\033[0m"


# ABCN dev set
# python3 evaluate_correction.py
# CEFR_LEVEL_FILENAME = "ABCN.dev.gold.bea19"
CEFR_LEVEL_FILENAME = "ABCN.dev.gold.bea19.first100"
TEST_FILE_PATH = f"test/{CEFR_LEVEL_FILENAME}.orig"
FINAL_OUTPUT_PATH = f"corrected_output/{CEFR_LEVEL_FILENAME}.corrected"
CSV_OUTPUT_PATH = f"corrected_output/{CEFR_LEVEL_FILENAME}.corrected.csv"
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")
LOCAL_ENDPOINT = os.getenv("LOCAL_ENDPOINT", "")
TOGETHER_ENDPOINT = os.getenv("TOGETHER_ENDPOINT", "")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
API_KEY = os.getenv("OPENAI_API_KEY", "")
MAX_RETRIES = 3  # Maximum number of retries for an API call
RETRY_DELAY = 30  # Delay in seconds before retrying an API
QPM_LIMIT = 3  # Queries per minute limit


# ABCN test set (evaluate on: https://codalab.lisn.upsaclay.fr/competitions/4057)
# TEST_FILE_PATH = "test/ABCN.test.bea19.orig"
# FINAL_OUTPUT_PATH = "corrected_output/ABCN.test.bea19.corrected"
# CSV_OUTPUT_PATH = "corrected_output/ABCN.test.bea19.corrected.csv"
# LOGGING_OUTPUT_PATH = "logs/processing.log"
# ERROR_OUTPUT_PATH = "logs/error.log"
# AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")
# API_KEY = os.getenv("OPENAI_API_KEY", "")
# MAX_RETRIES = 3  # Maximum number of retries for an API call
# RETRY_DELAY = 2  # Delay in seconds before retrying an API
# QPM_LIMIT = 3  # Queries per minute limit


# MODEL_NAME = "gpt-3.5-turbo"
# MODEL_NAME = "gpt-3.5-turbo-1106"
# MODEL_NAME = "llama-2-7b-chat.Q8_0.gguf"
# MODEL_NAME = "gpt-4-1106-preview"
# MODEL_NAME = "togethercomputer/Llama-2-7B-32K-Instruct"
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"


# GRAMMAR_PROMPT = """You are a language model assistant specialized in grammatical error correction. Your task is to:

# 1. Identify and correct grammatical errors in the user-provided text. Focus on fixing issues related to verb tense, subject-verb agreement, pronoun usage, article application, and other grammatical inaccuracies to ensure the text adheres to standard English grammar rules.

# Please return the grammatically corrected text in the following JSON format for clarity and further processing:

# {"text": "Your grammatically corrected text here."}

# For example, if the input is "Travel by bus is exspensive , bored and annoying .", your output should be:

# {"text": "Travelling by bus is expensive , boring and annoying ."}

# Note: Your primary objective is to correct grammatical errors. Do not focus on enhancing clarity or flow unless it directly pertains to correcting a grammatical mistake. The output will be evaluated using the ERRANT scorer, which focuses on the grammatical accuracy of the corrections."""


# GRAMMAR_PROMPT = """You are a language model assistant specialized in grammatical error correction. Your task is to:
# 1. Identify and correct grammatical errors in the user-provided text. Focus on fixing issues related to verb tense, subject-verb agreement, pronoun usage, article application, and other grammatical inaccuracies to ensure the text adheres to standard English grammar rules.
# Return the grammatically corrected text in the JSON format, without any explanatory text.

# # Desired format
# For example, if the input is "Travel by bus is exspensive , bored and annoying .", your output should be JSON only:
# {"text": "Travelling by bus is expensive, boring and annoying."}

# Note: The output will be evaluated using the ERRANT scorer, which focuses on the grammatical accuracy of the corrections."""

GRAMMAR_PROMPT = """You are a language model assistant specialized in grammatical error correction. Your task is to:
1. Identify and correct grammatical errors in the user-provided text. Focus on fixing issues related to verb tense, subject-verb agreement, pronoun usage, article application, and other grammatical inaccuracies to ensure the text adheres to standard English grammar rules.
Return the grammatically corrected text in the JSON format, without any explanatory text.

# Desired format
For example, if the input is "Travel by bus is exspensive , bored and annoying .", your output should be JSON only:
{"text": "Travelling by bus is expensive, boring and annoying."}

Note: The output will be evaluated using the ERRANT scorer, which focuses on the grammatical accuracy of the corrections."""


# Generate a unique identifier for this run based on the current timestamp
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Define log file paths with the unique run identifier
LOGGING_OUTPUT_PATH = f"logs/run_{run_id}.log"
ERROR_OUTPUT_PATH = f"logs/error_{run_id}.log"


# Configure logging to output to a file
logging.basicConfig(
    level=logging.INFO,
    format=f"{BLUE}%(asctime)s{RESET} - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOGGING_OUTPUT_PATH),
        logging.StreamHandler(),
    ],
)

# Create a separate handler for error logs
error_handler = logging.FileHandler(ERROR_OUTPUT_PATH)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(
    logging.Formatter(f"{RED}%(asctime)s{RESET} - %(levelname)s - %(message)s")
)


# Get the root logger and add the error handler
root_logger = logging.getLogger()
root_logger.addHandler(error_handler)


# Initialize the OpenAI client based on the selected model
def get_openai_client(model_name: str) -> Any:
    if model_name == "llama-2-7b-chat.Q8_0.gguf":
        # Point to the local server
        return openai.AsyncOpenAI(
            base_url=LOCAL_ENDPOINT, api_key="not-needed"
        )
    if model_name in [
        "togethercomputer/Llama-2-7B-32K-Instruct",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
    ]:
        # Point to the local server
        return openai.AsyncOpenAI(
            base_url=TOGETHER_ENDPOINT, api_key=TOGETHER_API_KEY
        )
    else:
        # Initialize the OpenAI client with Azure endpoint and API key
        return openai.AsyncAzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_version="2023-12-01-preview",
            api_key=API_KEY,
        )


client = get_openai_client(MODEL_NAME)


# Rate limiter using an asyncio Semaphore
class RateLimiter:
    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.semaphore = asyncio.Semaphore(rate_limit)

    async def __aenter__(self):
        await self.semaphore.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await asyncio.sleep(60 / self.rate_limit)
        self.semaphore.release()


rate_limiter = RateLimiter(QPM_LIMIT)


def format_user_content(text: str) -> str:
    return json.dumps({"input": text})


async def ask_llm(
    client: Any,
    prompt: str,
    text: str,
    line_number: int,
    total_lines: int,
    model_name: str,
) -> str:
    retries = 0
    while retries < MAX_RETRIES:
        try:
            logging.info(
                f"Sending request for line {line_number}/{total_lines}: {text}"
            )

            # TODO: refactor
            model_params = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": format_user_content(text)},
                ],
                "temperature": 0,
            }

            if model_name in ["gpt-4-1106-preview", "gpt-3.5-turbo-1106"]:
                model_params["response_format"] = {"type": "json_object"}

            completion = await client.chat.completions.create(**model_params)

            # Parse the 'content' field as JSON
            response = completion.choices[0].message.content
            logging.info(
                f"{YELLOW}Received raw response for line {line_number}/{total_lines}: {response}{RESET}"
            )
            content_json = json.loads(response)
            response_text = content_json.get("text")
            if response_text is not None:
                return response_text
            else:
                # If 'text' field is not present, log and retry
                logging.warning(
                    f"'text' field not found in response JSON for line {line_number}/{total_lines}. Retrying..."
                )
                retries += 1
                await asyncio.sleep(RETRY_DELAY)  # Wait before retrying
        except json.JSONDecodeError as e:
            logging.error(
                f"JSON parsing error for line {line_number}/{total_lines}: {e}"
            )
            retries += 1
            await asyncio.sleep(RETRY_DELAY)  # Wait before retrying
        except Exception as e:
            logging.error(
                f"An error occurred while processing line {line_number}/{total_lines}: {e}"
            )
            retries += 1
            logging.info(
                f"{YELLOW}Retrying for line {line_number}/{total_lines} (Attempt {retries}/{MAX_RETRIES}){RESET}"
            )
            await asyncio.sleep(RETRY_DELAY)  # Wait before retrying

    logging.error(
        f"Max retries reached for line {line_number}/{total_lines}. Exiting the program."
    )
    sys.exit(
        1
    )  # Exit the program with a non-zero exit code to indicate an error


async def correct_grammar_and_write_csv(
    client: Any,
    text: str,
    line_number: int,
    total_lines: int,
    csv_writer: Any,
    model_name: str,
) -> str:
    async with rate_limiter:
        corrected_text = await ask_llm(
            client, GRAMMAR_PROMPT, text, line_number, total_lines, model_name
        )
        # Process the corrected text with spaCy
        doc = nlp(corrected_text.strip())
        processed_text = " ".join(token.text for token in doc)
        logging.info(
            f"{GREEN}Received correction for line {line_number}/{total_lines}: {processed_text}{RESET}"
        )
        # Create a dictionary for the CSV row
        row = {
            "Line Number": line_number,
            "Original Sentence": text.strip(),
            "Corrected Sentence": processed_text,
        }
        # Use the writerow method to write the dictionary to the CSV
        await csv_writer.writerow(row)
        return processed_text


# Function to check which lines have already been processed
async def get_processed_lines(csv_output_path: str) -> set[int]:
    processed_lines = set()
    try:
        async with aiofiles.open(csv_output_path, "r") as csv_file:
            content = await csv_file.read()
            reader = DictReader(content.splitlines())
            for row in reader:
                try:
                    line_number = int(row["Line Number"])
                    processed_lines.add(line_number)
                except (ValueError, KeyError):
                    # Skip rows with invalid or missing "Line Number"
                    continue
    except FileNotFoundError:
        # If the CSV file does not exist, return an empty set
        pass
    return processed_lines


async def process_file(client: Any, test_file_path: str, csv_output_path: str):
    processed_lines = await get_processed_lines(csv_output_path)

    # Check if the file exists and has more than just the header
    file_exists = os.path.exists(csv_output_path)
    should_write_header = (
        not file_exists or os.stat(csv_output_path).st_size == 0
    )

    async with aiofiles.open(test_file_path, "r") as test_file, aiofiles.open(
        csv_output_path, "a", newline=""
    ) as csv_file:
        lines = await test_file.readlines()
        csv_writer = DictWriter(
            csv_file,
            fieldnames=[
                "Line Number",
                "Original Sentence",
                "Corrected Sentence",
            ],
        )

        if should_write_header:
            # Write the header using aiofiles interface
            await csv_file.write(
                '"Line Number","Original Sentence","Corrected Sentence"\n'
            )

        tasks = [
            correct_grammar_and_write_csv(
                client, line, i + 1, len(lines), csv_writer, MODEL_NAME
            )
            for i, line in enumerate(lines)
            if i + 1 not in processed_lines
        ]
        await asyncio.gather(*tasks)


async def generate_corrected_file_from_csv(
    csv_output_path: str, output_path: str
):
    async with aiofiles.open(csv_output_path, mode="r") as csv_file:
        # Read the entire file into memory
        csv_content = await csv_file.read()

    # Parse the CSV content
    csv_reader = DictReader(csv_content.splitlines())

    # Sort the rows by 'Line Number' (convert to int for proper numeric sort)
    sorted_rows = sorted(csv_reader, key=lambda row: int(row["Line Number"]))

    # Write the sorted 'Corrected Sentence' lines to the output file
    async with aiofiles.open(output_path, mode="w") as output_file:
        for row in sorted_rows:
            if "Corrected Sentence" in row:
                await output_file.write(row["Corrected Sentence"] + "\n")
            else:
                logging.warning(
                    f"Key 'Corrected Sentence' not found in row: {row}"
                )


# Function to log a divider when the program exits
def log_exit_divider():
    logging.info("=" * 80)


# Register the exit function
atexit.register(log_exit_divider)

# Run the script
if __name__ == "__main__":
    logging.info("=" * 80)
    logging.info("Starting to process the file...")
    asyncio.run(process_file(client, TEST_FILE_PATH, CSV_OUTPUT_PATH))
    asyncio.run(
        generate_corrected_file_from_csv(CSV_OUTPUT_PATH, FINAL_OUTPUT_PATH)
    )
    logging.info("File processing completed.")
    logging.info("=" * 80)
