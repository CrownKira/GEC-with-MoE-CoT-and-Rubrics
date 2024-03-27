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
import subprocess
import groq


# python3 main_batch.py
# python3 commands/evaluate_correction.py


# Load the spaCy model outside of the asynchronous function to avoid reloading it multiple times
nlp = spacy.load("en_core_web_sm")

# Load environment variables from .env file
load_dotenv()


# TUNABLE CONFIGS

# CONFIGS: MODEL
OPENAI_MODELS = [
    "gpt-3.5-turbo",
]

OPENAI_JSON_MODE_SUPPORTED_MODELS = [
    "gpt-3.5-turbo-1106",
    "gpt-4-1106-preview",
]

LOCAL_LLM_MODELS = [
    "llama-2-7b-chat.Q8_0.gguf",
]

TOGETHER_AI_MODELS = [
    "togethercomputer/Llama-2-7B-32K-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]

GROQ_MODELS = [
    "gemma-7b-it",
    "llama2-70b-4096",
    "mixtral-8x7b-32768",
]

# MODEL_NAME = OPENAI_JSON_MODE_SUPPORTED_MODELS[0]  # gpt-4-1106-preview
# MODEL_NAME = TOGETHER_AI_MODELS[1]
MODEL_NAME = GROQ_MODELS[2]


# CONFIGS: PROMPT
# GRAMMAR_VARIANT = "standard American"
GRAMMAR_VARIANT = "British"
NEXT_TOKEN = "<|NEXT|>\n"


# CONFIGS: RAG


# NON-TUNABLE CONFIGS

# CONFIGS: INPUT PREPROCESSING
MAX_TOKENS = 1024
BATCH_SIZE_IN_TOKENS = int(MAX_TOKENS * 0.7)
# CHUNK_OVERLAP_IN_TOKENS = 50


# CONFIGS: PATHS
# ABCN dev set
CEFR_LEVEL_FILENAME = "ABCN.dev.gold.bea19.first100"
TEST_FILE_PATH = f"test/{CEFR_LEVEL_FILENAME}.orig"
FINAL_OUTPUT_PATH = f"corrected_output/{CEFR_LEVEL_FILENAME}.corrected"
CSV_OUTPUT_PATH = f"corrected_output/{CEFR_LEVEL_FILENAME}.corrected.csv"


# CONFIGS: API
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")
LOCAL_ENDPOINT = os.getenv("LOCAL_ENDPOINT", "")
TOGETHER_ENDPOINT = os.getenv("TOGETHER_ENDPOINT", "")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
API_KEY = os.getenv("OPENAI_API_KEY", "")
MAX_RETRIES = 3  # Maximum number of retries for an API call
RETRY_DELAY = 30  # Delay in seconds before retrying an API
QPM_LIMIT = 3  # Queries per minute limit


# CONFIGS: OTHERS
# ANSI escape codes for colors
RED = "\033[1;31m"
GREEN = "\033[1;32m"
YELLOW = "\033[93m"
BLUE = "\033[1;34m"
RESET = "\033[0m"


# TODO: remind fix what issues later
# GRAMMAR_PROMPT = """You are a language model assistant specializing in grammatical error correction. Your tasks are to:
# 1. Identify and correct grammatical errors in the user-provided text. Focus on fixing issues related to verb tense, subject-verb agreement, pronoun usage, article application, and other grammatical inaccuracies to ensure the text adheres to {0} English grammar rules.
# 2. Maintain consistency in grammar correction (e.g., past or present tense) in parts of the input text that you think are contextually related.
# 3. Return the grammatically corrected text in JSON format, without any explanatory text.

# # Desired format
# For example, if the input is:
# {{"input": "Travel by bus is exspensive , bored and annoying .{1}I go to school yesterday .{1}He do not likes the food."}}

# Your output should be JSON only:
# {{"text": "Travelling by bus is expensive, boring, and annoying.{1}I went to school yesterday.{1}He does not like the food."}}

# Note: The output will be evaluated using the ERRANT scorer, which focuses on the grammatical accuracy of the corrections.""".format(
#     GRAMMAR_VARIANT, NEXT_TOKEN
# )


"""
TODO: add config vars 

base
grammar_variant
consistency_reminder
detailed_correction_focus
"""
GRAMMAR_PROMPT = """You are a language model assistant specializing in grammatical error correction. Your tasks are to:
1. Identify and correct grammatical errors in the user-provided text. Ensure the text adheres to {0} English grammar rules.
2. Maintain consistency in grammar correction (e.g., past or present tense) in adjacent lines of the input text that you think are contextually related.
3. Return the grammatically corrected text in JSON format, without any explanatory text.

# Desired format
For example, if the input is:
{{"input": "Yesterday, we goes to the local park.{1}It was very crowded, but we finds a quiet spot for a picnic.{1}Unfortunately, we forgets our picnic basket at home."}}

Your output should be JSON only:
{{"text": "Yesterday, we went to the local park.{1}It was very crowded, but we found a quiet spot for a picnic.{1}Unfortunately, we forgot our picnic basket at home."}}

Note: The output will be evaluated using the ERRANT scorer, which focuses on the grammatical accuracy of the corrections.""".format(
    GRAMMAR_VARIANT, NEXT_TOKEN
)

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
    if model_name in GROQ_MODELS:
        return groq.AsyncGroq(api_key=GROQ_API_KEY)
    if model_name in LOCAL_LLM_MODELS:
        # Point to the local server
        return openai.AsyncOpenAI(base_url=LOCAL_ENDPOINT, api_key="not-needed")
    if model_name in TOGETHER_AI_MODELS:
        # Point to the local server
        return openai.AsyncOpenAI(base_url=TOGETHER_ENDPOINT, api_key=TOGETHER_API_KEY)
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
    # TODO: better way?
    text_with_next_token = text.replace("\n", f"{NEXT_TOKEN}")
    return json.dumps({"input": text_with_next_token})


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

        # TODO: better way to not strip?
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
            if model_name in OPENAI_JSON_MODE_SUPPORTED_MODELS:
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

            # TODO: extract to a function
            corrected_lines = []
            for line in response_text.split(NEXT_TOKEN):
                corrected_lines.append(line.strip())

            final_text = "\n".join(corrected_lines)

            if len(corrected_lines) != len(text.split("\n")):
                print(
                    "check lines:",
                    len(corrected_lines),
                    len(text.split("\n")),
                )
                raise ValueError(
                    "Number of lines in response_text does not match the number of lines in text"
                )

            return final_text
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

        # TODO: better way?
        # Process the corrected text with spaCy
        doc = nlp(corrected_text.strip())
        processed_text = " ".join(token.text for token in doc)
        stripped_lines = [line.strip() for line in processed_text.split("\n")]
        processed_text = "\n".join(stripped_lines)

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

    # Check for existing output files
    if os.path.exists(FINAL_OUTPUT_PATH) or os.path.exists(CSV_OUTPUT_PATH):
        user_input = (
            input(
                "Existing output files found. Do you want to continue with existing files? Type 'reset' to delete and start fresh: "
            )
            .strip()
            .lower()
        )
        if user_input == "reset":
            if os.path.exists(FINAL_OUTPUT_PATH):
                os.remove(FINAL_OUTPUT_PATH)
            if os.path.exists(CSV_OUTPUT_PATH):
                os.remove(CSV_OUTPUT_PATH)
            print("Existing files removed. Starting fresh...")
        else:
            print("Continuing with existing files...")

    processed_batches = await get_processed_batches(csv_output_path)
    # Check if the file exists and has more than just the header
    file_exists = os.path.exists(csv_output_path)
    should_write_header = not file_exists or os.stat(csv_output_path).st_size == 0
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
    with open(csv_output_path, mode="r", newline="", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        sorted_rows = sorted(csv_reader, key=lambda row: int(row["Batch Number"]))

    with open(output_path, mode="w", newline="", encoding="utf-8") as output_file:
        for row in sorted_rows:
            if "Corrected Text" in row:
                corrected_lines = row["Corrected Text"].split("\n")
                for corrected_line in corrected_lines:
                    output_file.write(corrected_line + "\n")


# Function to log a divider when the program exits
def log_exit_divider():
    logging.info("=" * 80)


# Register the exit function
atexit.register(log_exit_divider)


def prompt_for_evaluation():
    user_response = (
        input("Do you want to evaluate the result? (yes/no): ").strip().lower()
    )
    if user_response == "yes":
        try:
            # Execute the evaluation script
            print("Evaluating the corrections...")
            subprocess.run(["python3", "commands/evaluate_correction.py"], check=True)
            print("Evaluation completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred during evaluation: {e}")
    elif user_response == "no":
        print("Evaluation skipped.")
    else:
        print("Invalid input. Please type 'yes' or 'no'.")
        prompt_for_evaluation()


if __name__ == "__main__":
    logging.info("=" * 80)
    logging.info(f"Model selected: {MODEL_NAME}")
    logging.info(f"{BLUE}Using prompt: {GRAMMAR_PROMPT}{RESET}")
    logging.info("Starting to process the file...")
    asyncio.run(process_file(client, TEST_FILE_PATH, CSV_OUTPUT_PATH))
    logging.info("Generating the corrected file from CSV...")
    generate_corrected_file_from_csv(CSV_OUTPUT_PATH, FINAL_OUTPUT_PATH)
    logging.info("File processing completed.")
    prompt_for_evaluation()
    logging.info("=" * 80)
