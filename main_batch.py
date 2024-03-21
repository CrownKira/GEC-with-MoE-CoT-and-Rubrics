import os
import sys
import json
import openai
import asyncio
import logging
import aiofiles
import csv
from dotenv import load_dotenv
import atexit
from typing import Any, List
import spacy
import logging
import datetime
from tiktoken import get_encoding


# Load the spaCy model outside of the asynchronous function to avoid reloading it multiple times
nlp = spacy.load("en_core_web_sm")

# Load environment variables from .env file
load_dotenv()

# Configuration variables
MAX_TOKENS = 1024
BATCH_SIZE_IN_TOKENS = int(MAX_TOKENS * 0.7)
# CHUNK_OVERLAP_IN_TOKENS = 50


# ANSI escape codes for colors
RED = "\033[1;31m"
GREEN = "\033[1;32m"
YELLOW = "\033[93m"
BLUE = "\033[1;34m"
RESET = "\033[0m"

# ABCN dev set
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

MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"

GRAMMAR_PROMPT = """You are a language model assistant specialized in grammatical error correction. Your task is to:
1. Identify and correct grammatical errors in the user-provided text. Focus on fixing issues related to verb tense, subject-verb agreement, pronoun usage, article application, and other grammatical inaccuracies to ensure the text adheres to standard English grammar rules.
Return the grammatically corrected text in the JSON format, without any explanatory text.

# Desired format
For example, if the input is:
Travel by bus is exspensive , bored and annoying .
I go to school yesterday .

Your output should be JSON only:
{"text": "Travelling by bus is expensive, boring and annoying.\nI went to school yesterday."}

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


def count_tokens(text: str) -> int:
    enc = get_encoding("gpt2")
    tokens = enc.encode(text)
    token_count = len(tokens)
    return token_count


def calculate_avg_chars_per_token(sample_text: str) -> float:
    total_tokens = count_tokens(sample_text)
    total_chars = len(sample_text)
    return total_chars / total_tokens


def split_text_into_batches(
    text: str,
    batch_size_in_tokens: int = 10,
) -> List[str]:
    lines = text.split("\n")
    batches = []
    current_batch = ""
    current_batch_tokens = 0

    for line in lines:
        line_tokens = count_tokens(
            line + "\n"
        )  # Include newline character in token count
        if line_tokens > batch_size_in_tokens:
            print(
                f"Error: Line exceeds the batch size of {batch_size_in_tokens} tokens."
            )
            print("Line:", line)
            print("Tokens:", line_tokens)
            sys.exit(1)

        if current_batch_tokens + line_tokens <= batch_size_in_tokens:
            current_batch += line + "\n"
            current_batch_tokens += line_tokens
        else:
            batches.append(current_batch.strip())
            current_batch = line + "\n"
            current_batch_tokens = line_tokens

    if current_batch.strip():
        batches.append(current_batch.strip())

    return batches


async def ask_llm(
    client: Any,
    prompt: str,
    text: str,
    batch_number: int,
    total_batches: int,
    model_name: str,
) -> str:
    retries = 0
    while retries < MAX_RETRIES:
        try:
            logging.info(
                f"Sending request for batch {batch_number}/{total_batches}: {text}"
            )
            model_params = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": format_user_content(text)},
                ],
                "temperature": 0,
                "max_tokens": MAX_TOKENS,
            }
            if model_name in ["gpt-4-1106-preview", "gpt-3.5-turbo-1106"]:
                model_params["response_format"] = {"type": "json_object"}
            completion = await client.chat.completions.create(**model_params)
            response = completion.choices[0].message.content
            logging.info(
                f"{YELLOW}Received raw response for batch {batch_number}/{total_batches}: {response}{RESET}"
            )
            content_json = json.loads(response)
            response_text = content_json.get("text")
            if response_text is None:
                raise ValueError("'text' field not found in response JSON")

            # TODO: fix this
            # if len(response_text.split("\n")) != len(text.split("\n")):
            #     print(
            #         "check lines:",
            #         len(response_text.split("\n")),
            #         len(text.split("\n")),
            #     )
            #     raise ValueError(
            #         "Number of lines in response_text does not match the number of lines in text"
            #     )

            return response_text
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(
                f"Error processing response for batch {batch_number}/{total_batches}: {e}"
            )
        except Exception as e:
            logging.error(
                f"An error occurred while processing batch {batch_number}/{total_batches}: {e}"
            )
        retries += 1
        if retries < MAX_RETRIES:
            logging.info(
                f"{YELLOW}Retrying for batch {batch_number}/{total_batches} (Attempt {retries}/{MAX_RETRIES}){RESET}"
            )
            await asyncio.sleep(RETRY_DELAY)
        else:
            logging.error(
                f"Max retries reached for batch {batch_number}/{total_batches}. Exiting the program."
            )
            sys.exit(1)  # Exit the program with a non-zero status code
    raise RuntimeError("Unexpected execution path")


async def correct_grammar_and_write_csv(
    client: Any,
    text: str,
    batch_number: int,
    total_batches: int,
    csv_writer: Any,
    model_name: str,
) -> str:
    async with rate_limiter:
        corrected_text = await ask_llm(
            client,
            GRAMMAR_PROMPT,
            text,
            batch_number,
            total_batches,
            model_name,
        )

        # Process the corrected text with spaCy
        doc = nlp(corrected_text.strip())
        processed_text = " ".join(token.text for token in doc)
        logging.info(
            f"{GREEN}Received correction for batch {batch_number}/{total_batches}: {processed_text}{RESET}"
        )
        # Write the batch number and corrected text to the CSV
        row = {
            "Batch Number": batch_number,
            "Corrected Text": processed_text,
        }
        await csv_writer.writerow(row)
        return processed_text


# Function to check which batches have already been processed
async def get_processed_batches(csv_output_path: str) -> set[int]:
    processed_batches = set()
    try:
        async with aiofiles.open(csv_output_path, "r") as csv_file:
            content = await csv_file.read()
            reader = csv.DictReader(content.splitlines())
            for row in reader:
                try:
                    batch_number = int(row["Batch Number"])
                    processed_batches.add(batch_number)
                except (ValueError, KeyError):
                    # Skip rows with invalid or missing "Batch Number"
                    continue
    except FileNotFoundError:
        # If the CSV file does not exist, return an empty set
        pass
    return processed_batches


async def process_file(client: Any, test_file_path: str, csv_output_path: str):
    processed_batches = await get_processed_batches(csv_output_path)
    # Check if the file exists and has more than just the header
    file_exists = os.path.exists(csv_output_path)
    should_write_header = (
        not file_exists or os.stat(csv_output_path).st_size == 0
    )
    async with aiofiles.open(test_file_path, "r") as test_file, aiofiles.open(
        csv_output_path, "a", newline=""
    ) as csv_file:
        text = await test_file.read()
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "Batch Number",
                "Corrected Text",
            ],
        )
        if should_write_header:
            await csv_file.write('"Batch Number","Corrected Text"\n')
        batches = split_text_into_batches(text, BATCH_SIZE_IN_TOKENS)
        total_batches = len(batches)
        tasks = []
        for batch_number, batch_text in enumerate(batches, start=1):
            if batch_number in processed_batches:
                continue
            tasks.append(
                correct_grammar_and_write_csv(
                    client,
                    batch_text,
                    batch_number,
                    total_batches,
                    csv_writer,
                    MODEL_NAME,
                )
            )
        await asyncio.gather(*tasks)


def generate_corrected_file_from_csv(csv_output_path: str, output_path: str):
    with open(
        csv_output_path, mode="r", newline="", encoding="utf-8"
    ) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        sorted_rows = sorted(
            csv_reader, key=lambda row: int(row["Batch Number"])
        )

    with open(
        output_path, mode="w", newline="", encoding="utf-8"
    ) as output_file:
        for row in sorted_rows:
            if "Corrected Text" in row:
                # Ensure to process and write the "Corrected Text" as needed
                corrected_lines = row["Corrected Text"].split("\n")
                for corrected_line in corrected_lines:
                    output_file.write(corrected_line.strip() + "\n")


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
    generate_corrected_file_from_csv(CSV_OUTPUT_PATH, FINAL_OUTPUT_PATH)
    logging.info("File processing completed.")
    logging.info("=" * 80)
