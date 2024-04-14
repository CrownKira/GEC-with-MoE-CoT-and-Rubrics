import os
import sys
import json
import openai
import asyncio
import aiofiles
import csv
from dotenv import load_dotenv
import atexit
from typing import Any, List, Optional, Callable, Dict
import spacy
import logging
import datetime
from tiktoken import get_encoding
import subprocess
import groq
from clients.coze import AsyncCoze
import spacy
import errant
import argparse
import json
import logging
from typing import Any, List, Optional, Callable


# python3 -m systems.greco2 $'This first sentence is.\nSecond is sentence.\nThird the is sentence.' --quiet
# python3 -m systems.greco2 $'This first sentence is.\nSecond is sentence.\nThird the is sentence.'
# python3 -m systems.greco2 $'It \'s difficult answer at the question \" what are you going to do in the future ? \" if the only one who has to know it is in two minds .\nWhen I was younger I used to say that I wanted to be a teacher , a saleswoman and even a butcher .. I do n\'t know why .\nI would like to study Psychology because one day I would open my own psychology office and help people .\nIt \'s difficult because I \'ll have to study hard and a lot , but I think that if you like a subject , you \'ll study it easier .\nMaybe I \'ll change my mind , maybe not .\nI think that the public transport will always be in the future .\nThe rich people will buy a car but the poor people always need to use a bus or taxi .\nI consider that is more convenient to drive a car because you carry on more things in your own car than travelling by car .\nAlso , you \'ll meet friendly people who usually ask to you something to be friends and change your telephone number .\nIn my experience when I did n\'t have a car I used to use the bus to go to the school and go back to my house .\nIn my opinion , the car is n\'t necessary when you have crashed in the street , in that moment you realized the importance of a public transport .\nIn India we have various types of Public transport , like Cycle , Bike , Car , Train & Flight .\nDepending on the distance and duration to the desired place , mode of transport is chosen accordingly .\nBut Generally speaking , travelling by car is much more fun when compared with other modes of transport .\nThis reminds me of a trip that I have recently been to and the place is Agra .\nIt takes around 6 hours by National highway to go from Delhi to Agra .\nWe have stopped at hotels for having food and just in case if any of us feels hungry , we have purchased some snacks just before the trip .\nSince , we have the option to wait anytime we want to when we travel by car ( which is impossible when travelling by train & Flight ) .\nIn addition to it , we can also take a comfortable short nap on the back seat and wake up fresh .\nDue to the above mentioned reasons , I am going to conclude that travelling by car is much more convenient .\nMy name is Sarah .\nI am 17 years old .\nI am looking forward to join you in this year summer camps .\nI love children , and I enjoy looking after them . also , I organized many sports activities before in my school .\nIn addition to that , i enjoy cooking .\nMy family think that my cook is amazing .\nI hope that you give my the chance to join you .\nThanks\nMy favourite sport is volleyball because I love plays with my friends .'
# python3 -m systems.greco2 $'When I was younger I used to say that I wanted to be a teacher , a saleswoman and even a butcher .. I do n\'t know why .\nI would like to study Psychology because one day I would open my own psychology office and help people .\nIt \'s difficult because I \'ll have to study hard and a lot , but I think that if you like a subject , you \'ll study it easier .\nMaybe I \'ll change my mind , maybe not .\nI think that the public transport will always be in the future .\nThe rich people will buy a car but the poor people always need to use a bus or taxi .\nI consider that is more convenient to drive a car because you carry on more things in your own car than travelling by car .\nAlso , you \'ll meet friendly people who usually ask to you something to be friends and change your telephone number .\nIn my experience when I did n\'t have a car I used to use the bus to go to the school and go back to my house .\nIn my opinion , the car is n\'t necessary when you have crashed in the street , in that moment you realized the importance of a public transport .\nIn India we have various types of Public transport , like Cycle , Bike , Car , Train & Flight .\nDepending on the distance and duration to the desired place , mode of transport is chosen accordingly .\nBut Generally speaking , travelling by car is much more fun when compared with other modes of transport .\nThis reminds me of a trip that I have recently been to and the place is Agra .\nIt takes around 6 hours by National highway to go from Delhi to Agra .\nWe have stopped at hotels for having food and just in case if any of us feels hungry , we have purchased some snacks just before the trip .\nSince , we have the option to wait anytime we want to when we travel by car ( which is impossible when travelling by train & Flight ) .\nIn addition to it , we can also take a comfortable short nap on the back seat and wake up fresh .\nDue to the above mentioned reasons , I am going to conclude that travelling by car is much more convenient .\nMy name is Sarah .\nI am 17 years old .\nI am looking forward to join you in this year summer camps .\nI love children , and I enjoy looking after them . also , I organized many sports activities before in my school .\nIn addition to that , i enjoy cooking .\nMy family think that my cook is amazing .\nI hope that you give my the chance to join you .\nThanks\nMy favourite sport is volleyball because I love plays with my friends .'


# Ensure you have loaded the spaCy model at the start of your script
nlp = spacy.load("en_core_web_sm")
annotator = errant.load("en")


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

# coze bot ids
COZE_BOTS = [
    "7351253103510978578",
]


QUALITY_ESTIMATION_MODEL_NAME = OPENAI_JSON_MODE_SUPPORTED_MODELS[0]


# change model here
# MODEL_NAME = OPENAI_JSON_MODE_SUPPORTED_MODELS[0]
# MODEL_NAME = TOGETHER_AI_MODELS[1]
# MODEL_NAME = GROQ_MODELS[2]
# MODEL_NAME = COZE_BOTS[0]


# CONFIGS: PROMPT
# GRAMMAR_VARIANT = "standard American"
GRAMMAR_VARIANT = "British"


# TEXT_DELIMITER = "|||"
TEXT_DELIMITER = "~~~"


# CONFIGS: RAG


# NON-TUNABLE CONFIGS


# CONFIGS: INPUT PREPROCESSING
# The maximum context length for the Azure GPT-3.5-turbo-1106 model is 16,385 tokens, which encompasses both input and output tokens. However, the limit for the output tokens specifically is set at 4,096 tokens. When calling the API, you should ensure that max_tokens <= 4096 and the sum of input_tokens + max_tokens <= 16385​ (OpenAI Developer Forum)​.
# DEFAULT_MAX_TOKENS = 1024
# DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.1
DEFAULT_FREQUENCY_PENALTY = 0
# MAX_TOKENS = 1024
# QUALITY_ESTIMATION_FREQUENCY_PENALTY = 0.2
# QUALITY_ESTIMATION_FREQUENCY_PENALTY = 0.1
QUALITY_ESTIMATION_FREQUENCY_PENALTY = 0
# BATCH_SIZE_IN_TOKENS = int(MAX_TOKENS * 0.6)
VOTE_INCREASE_FACTOR = 0.05
MAX_SCORE_CAP = 110  # Maximum allowed score
# CHUNK_OVERLAP_IN_TOKENS = 50


# CONFIGS: PATHS
# ABCN dev set
CEFR_LEVEL_FILENAME = "ABCN.dev.gold.bea19.first5"
TEST_FILE_PATH = f"test/{CEFR_LEVEL_FILENAME}.orig"
FINAL_OUTPUT_PATH = f"corrected_output/{CEFR_LEVEL_FILENAME}.corrected"
CSV_OUTPUT_PATH = f"corrected_output/{CEFR_LEVEL_FILENAME}.corrected.csv"


# CONFIGS: API
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")
LOCAL_ENDPOINT = os.getenv("LOCAL_ENDPOINT", "")
TOGETHER_ENDPOINT = os.getenv("TOGETHER_ENDPOINT", "")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COZE_API_KEY = os.getenv("COZE_API_KEY", "")
RETRY_DELAY = 5  # Delay in seconds before retrying an API
QPM_LIMIT = 5  # Queries per minute limit
# MAX_RETRIES = 3  # Maximum number of retries for an API call
MAX_RETRIES = 30  # Maximum number of iterations to attempt to complete JSON or due to other retry conditions
CONTINUE_PROMPT = "Continue to complete the JSON above."


# CONFIGS: OTHERS
# ANSI escape codes for colors
RED = "\033[1;31m"
GREEN = "\033[1;32m"
YELLOW = "\033[93m"
BLUE = "\033[1;34m"
RESET = "\033[0m"


# TODO: explain before answer; give more examples; etc

QUALITY_ESTIMATION_PROMPT = """You are an AI specialized in assessing the quality of grammatical error corrections from JSON inputs. Given an input JSON with 'original' and 'corrected' keys containing lists of sentences, evaluate each corrected sentence based on:
1. Correction of spelling mistakes, punctuation errors, verb tense issues, word choice problems, and other grammatical mistakes.
2. Preservation of the original meaning, penalizing deviations.
3. Appropriateness of language in context.

Please rate each correction on a scale from 0 to 100 and return your evaluations in JSON format as follows:
{{"scores": ["sentence 1 score", "sentence 2 score", ...]}}

# Example
If the input JSON is:
{{
  "original": ["He go to school every day.", "She walk to the park."],
  "corrected": ["He goes to school every day.", "She walks to the park."]
}}
Your output should be JSON only, like:
{{"scores": [95, 90]}}

Ensure that the number of scores matches the number of corrected sentences provided.
"""


# QUALITY_ESTIMATION_PROMPT_V2 = """You are an English teacher who assesses the quality of students' grammatical error corrections. Evaluate each student's corrected sentence based on the criteria outlined in the rubric below:

# ### Rubric for Evaluating Sentence Corrections:

# 1. **Spelling Errors [SPELL]**:
#    - Minor: -2 points [SPELL-MIN]
#    - Major: -5 points [SPELL-MAJ]

# 2. **Punctuation Errors [PUNCT]**:
#    - Minor: -2 points [PUNCT-MIN]
#    - Major: -5 points [PUNCT-MAJ]

# 3. **Verb Tense and Grammatical Accuracy [GRAM]**:
#    - Minor: -3 points [GRAM-MIN]
#    - Major: -7 points [GRAM-MAJ]

# 4. **Preservation of Meaning [MEAN]**:
#    - Minor deviations: -4 points [MEAN-MIN]
#    - Major alterations: -10 points [MEAN-MAJ]

# 5. **Language Appropriateness [LANG]**:
#    - Tone mismatch: -5 points [LANG-TONE]
#    - Incorrect formality level: -6 points [LANG-FORM]
#    - Mixing grammar variants: -3 points [LANG-MIX]
#    - Inappropriate vocabulary: -8 points [LANG-VOCAB]

# 6. **Others [OTHER]**:
#    - Any errors not covered above: Up to -10 points (at grader's discretion)

# For each identified error, provide feedback using the format: "[Subtag] [Explanation of the error and what would be a correct approach] [-Deduction]."

# # Desired Output JSON Format:
# Your feedback should categorize errors under specific tags and subtags, providing a detailed explanation for each error detected. Summarize the deductions and final scores in a structured report, formatted as JSON:

# {
#     "evaluations": [
#         {
#             "student_sentence": "In the midst of the storm, a ship was sailing in the open sea. It's crew, seasoned and resilient, were unphased by the brewing tempest.",
#             "student_sentence_feedback": [
#                 {
#                     "type": "SPELL-MIN",
#                     "description": "'It's crew' should be 'Its crew' to denote possession, not contraction",
#                     "deduction": -2
#                 },
#                 {
#                     "type": "GRAM-MIN",
#                     "description": "'were unphased' should be 'were unfazed' to correct the spelling mistake",
#                     "deduction": -2
#                 }
#             ],
#             "corrected_sentence": "In the midst of the storm, a ship was sailing in the open sea. Its crew, seasoned and resilient, were unfazed by the brewing tempest.",
#             "total_deductions": -4,
#             "score": 96
#         },
#         {
#             "student_sentence": "Their captain, a venerable seafarer known for hes bravery and wisdom, was steering the ship with a steady hand.",
#             "student_sentence_feedback": [
#                 {
#                     "type": "WORD",
#                     "description": "'hes bravery' should be 'his bravery' to correct the pronoun error",
#                     "deduction": -2
#                 },
#                 {
#                     "type": "GRAM-MIN",
#                     "description": "'venerable seafarer known for hes bravery' could be rephrased to 'venerable seafarer, known for his bravery,' for better clarity and use of commas",
#                     "deduction": -3
#                 }
#             ],
#             "corrected_sentence": "Their captain, a venerable seafarer known for his bravery and wisdom, was steering the ship with a steady hand.",
#             "total_deductions": -5,
#             "score": 95
#         },
#         {
#             "student_sentence": "Suddenly, a gigantic wave, unlike any they had seen before, approached. It’s size and ferocity could spell doom for them.",
#             "student_sentence_feedback": [
#                 {
#                     "type": "SPELL-MIN",
#                     "description": "'It’s size and ferocity' should be 'Its size and ferocity' to correctly use the possessive form",
#                     "deduction": -2
#                 },
#                 {
#                     "type": "PUNCT-MIN",
#                     "description": "'a gigantic wave, unlike any they had seen before, approached' – consider adding a semicolon before 'unlike' for stylistic emphasis and clarity",
#                     "deduction": -1
#                 }
#             ],
#             "corrected_sentence": "Suddenly, a gigantic wave, unlike any they had seen before, approached; its size and ferocity could spell doom for them.",
#             "total_deductions": -3,
#             "score": 97
#         },
#         {
#             "student_sentence": "The captain, realizing the gravity of their situation, ordered for the sails to be lowered. 'We must not underestemate this storm,' he declared.",
#             "student_sentence_feedback": [
#                 {
#                     "type": "WORD",
#                     "description": "'ordered for the sails to be lowered' should be 'ordered the sails to be lowered' to streamline the command",
#                     "deduction": -2
#                 },
#                 {
#                     "type": "TONE",
#                     "description": "'We must not underestemate this storm,' he declared – 'underestemate' should be 'underestimate'. Additionally, using 'he declared' after a direct command might be redundant. A more nuanced approach could enhance the dramatic tone; consider 'he proclaimed' for variation",
#                     "deduction": -3
#                 }
#             ],
#             "corrected_sentence": "The captain, realizing the gravity of their situation, ordered the sails to be lowered. 'We must not underestimate this storm,' he proclaimed.",
#             "total_deductions": -5,
#             "score": 95
#         }
#     ]
# }

# Ensure that the number of scores matches the number of corrected sentences provided.
# Your comprehensive feedback will guide improvements in grammatical accuracy and narrative consistency. Please ensure that each sentence is evaluated not only on its own merits but also in the context of the surrounding narrative.
# """


QUALITY_ESTIMATION_PROMPT_V2 = """You are an English teacher who assesses the quality of students' grammatical error corrections. Evaluate each student's corrected sentence based on the criteria outlined in the rubric below:

### Rubric for Evaluating Sentence Corrections:

1. **Spelling Errors [SPELL]**:
   - Minor: -2 points [SPELL-MIN]
   - Major: -5 points [SPELL-MAJ]

2. **Punctuation Errors [PUNCT]**:
   - Minor: -2 points [PUNCT-MIN]
   - Major: -5 points [PUNCT-MAJ]

3. **Verb Tense and Grammatical Accuracy [GRAM]**:
   - Minor: -3 points [GRAM-MIN]
   - Major: -7 points [GRAM-MAJ]

4. **Preservation of Meaning [MEAN]**:
   - Minor deviations: -4 points [MEAN-MIN]
   - Major alterations: -10 points [MEAN-MAJ]

5. **Language Appropriateness [LANG]**:
   - Tone mismatch: -5 points [LANG-TONE]
   - Incorrect formality level: -6 points [LANG-FORM]
   - Mixing grammar variants: -3 points [LANG-MIX]
   - Inappropriate vocabulary: -8 points [LANG-VOCAB]

6. **Others [OTHER]**:
   - Any errors not covered above: Up to -10 points (at grader's discretion)

For each identified error, provide feedback using the format: "[Subtag] [Explanation of the error and what would be a correct approach] [-Deduction]."

# Desired Output JSON Format:
Your feedback should categorize errors under specific tags and subtags, providing a detailed explanation for each error detected. Summarize the deductions and final scores in a structured report, formatted as JSON: 

{
    "total_student_sentences": 4,
    "evaluations": [
        {
            "unique_index": 0,
            "student_sentence": "In the midst of the storm, a ship was sailing in the open sea. It's crew, seasoned and resilient, were unphased by the brewing tempest.",
            "student_sentence_feedback": [
                {
                    "type": "SPELL-MIN",
                    "description": "'It's crew' should be 'Its crew' to denote possession, not contraction",
                    "deduction": -2
                },
                {
                    "type": "GRAM-MIN",
                    "description": "'were unphased' should be 'were unfazed' to correct the spelling mistake",
                    "deduction": -2
                }
            ],
            "corrected_sentence": "In the midst of the storm, a ship was sailing in the open sea. Its crew, seasoned and resilient, were unfazed by the brewing tempest.",
            "total_deductions": -4,
            "score": 96
        },
        {
            "unique_index": 1,
            "student_sentence": "Their captain, a venerable seafarer known for hes bravery and wisdom, was steering the ship with a steady hand.",
            "student_sentence_feedback": [
                {
                    "type": "WORD",
                    "description": "'hes bravery' should be 'his bravery' to correct the pronoun error",
                    "deduction": -2
                },
                {
                    "type": "GRAM-MIN",
                    "description": "'venerable seafarer known for hes bravery' could be rephrased to 'venerable seafarer, known for his bravery,' for better clarity and use of commas",
                    "deduction": -3
                }
            ],
            "corrected_sentence": "Their captain, a venerable seafarer known for his bravery and wisdom, was steering the ship with a steady hand.",
            "total_deductions": -5,
            "score": 95
        },
        {
            "unique_index": 2,
            "student_sentence": "Suddenly, a gigantic wave, unlike any they had seen before, approached. It’s size and ferocity could spell doom for them.",
            "student_sentence_feedback": [
                {
                    "type": "SPELL-MIN",
                    "description": "'It’s size and ferocity' should be 'Its size and ferocity' to correctly use the possessive form",
                    "deduction": -2
                },
                {
                    "type": "PUNCT-MIN",
                    "description": "'a gigantic wave, unlike any they had seen before, approached' – consider adding a semicolon before 'unlike' for stylistic emphasis and clarity",
                    "deduction": -1
                }
            ],
            "corrected_sentence": "Suddenly, a gigantic wave, unlike any they had seen before, approached; its size and ferocity could spell doom for them.",
            "total_deductions": -3,
            "score": 97
        },
        {
            "unique_index": 3,
            "student_sentence": "The captain, realizing the gravity of their situation, ordered for the sails to be lowered. 'We must not underestemate this storm,' he declared.",
            "student_sentence_feedback": [
                {
                    "type": "WORD",
                    "description": "'ordered for the sails to be lowered' should be 'ordered the sails to be lowered' to streamline the command",
                    "deduction": -2
                },
                {
                    "type": "TONE",
                    "description": "'We must not underestemate this storm,' he declared – 'underestemate' should be 'underestimate'. Additionally, using 'he declared' after a direct command might be redundant. A more nuanced approach could enhance the dramatic tone; consider 'he proclaimed' for variation",
                    "deduction": -3
                }
            ],
            "corrected_sentence": "The captain, realizing the gravity of their situation, ordered the sails to be lowered. 'We must not underestimate this storm,' he proclaimed.",
            "total_deductions": -5,
            "score": 95
        }
    ]
}

Ensure that the number of scores matches the number of corrected sentences provided.
Your comprehensive feedback will guide improvements in grammatical accuracy and narrative consistency. Please ensure that each sentence is evaluated not only on its own merits but also in the context of the surrounding narrative.
"""


GRAMMAR_PROMPT = """You are a language model assistant specializing in grammatical error correction. Your tasks are to:
1. Identify and correct grammatical errors in the user-provided text. Ensure the text adheres to {0} English grammar rules.
2. Maintain consistency in grammar correction (e.g., past or present tense) in adjacent lines of the input text that you think are contextually related.
3. Crucially, splitting the corrected text using the specified text delimiter, "{1}", whenever it appears in the input text. This division must be reflected in your output.
4. Returning the grammatically corrected text in JSON format, exclusively, without any supplementary explanatory text.

# Desired format
For example, if the input is:
{{"input": "Yesterday, we goes to the local park.{1}It was very crowded, but we finds a quiet spot for a picnic.{1}Unfortunately, we forgets our picnic basket at home."}}

Your output should be JSON only:
{{"text": "Yesterday, we went to the local park.{1}It was very crowded, but we found a quiet spot for a picnic.{1}Unfortunately, we forgot our picnic basket at home."}}

Note: The output will be evaluated using the ERRANT scorer, which focuses on the grammatical accuracy of the corrections.""".format(
    GRAMMAR_VARIANT, TEXT_DELIMITER
)


# Generate a unique identifier for this run based on the current timestamp
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Define log file paths with the unique run identifier
LOGGING_OUTPUT_PATH = f"logs/run_{run_id}.log"
ERROR_OUTPUT_PATH = f"logs/error_{run_id}.log"


# Configure logging to output to a file
parser = argparse.ArgumentParser(description="Process some inputs.")
parser.add_argument(
    "input_text",
    nargs="?",
    default=None,
    help="The input text to process. Optional.",
)
parser.add_argument(
    "--quiet",
    action="store_true",
    help="Run in quiet mode, producing only the final output.",
)
args = parser.parse_args()


if args.quiet:
    # Configure logging to exclude stdout for quiet mode
    logging.basicConfig(
        level=logging.INFO,
        format=f"{BLUE}%(asctime)s{RESET} - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOGGING_OUTPUT_PATH),
        ],
    )
else:
    # Existing logging configuration that includes stdout
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
# TODO: return type
def get_openai_client(model_name: str) -> Any:
    if model_name in GROQ_MODELS:
        return groq.AsyncGroq(api_key=GROQ_API_KEY)
    if model_name in LOCAL_LLM_MODELS:
        # Point to the local server
        return openai.AsyncOpenAI(
            base_url=LOCAL_ENDPOINT, api_key="not-needed"
        )
    if model_name in TOGETHER_AI_MODELS:
        # Point to the local server
        return openai.AsyncOpenAI(
            base_url=TOGETHER_ENDPOINT, api_key=TOGETHER_API_KEY
        )
    if model_name in COZE_BOTS:
        return AsyncCoze(api_key=COZE_API_KEY)

    # Initialize the OpenAI client with Azure endpoint and API key
    return openai.AsyncAzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_version="2023-12-01-preview",
        api_key=OPENAI_API_KEY,
    )


class InputParser:
    @staticmethod
    def parse_input(input_string: str) -> List[str]:
        # Parse a string of \n separated sentences into a list
        return input_string.strip().split("\n")


# TODO: config model_id
class ModelIOParser:
    @staticmethod
    def parse_model_output(
        model_output: str, input_sentences: List[str]
    ) -> List[str]:
        try:
            text_data = json.loads(model_output)
            input_text = text_data["text"]
            sentences = input_text.split("~~~")

            expected_num_sentences = len(input_sentences)
            actual_num_sentences = len(sentences)

            if actual_num_sentences < expected_num_sentences:
                diff = expected_num_sentences - actual_num_sentences
                raise ValueError(
                    f"Insufficient number of lines. Expected at least {expected_num_sentences}, got {actual_num_sentences}. Difference: {diff}."
                )

            return sentences

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"Parsing failed due to {str(e)}")

    @staticmethod
    def prepare_model_input(input_sentences: List[str]) -> str:
        # Splitting the input text into lines, and then joining them with the custom delimiter
        joined_text = "~~~".join(input_sentences)
        # Wrapping the joined text in a dictionary and serializing it to a JSON string
        return json.dumps({"input": joined_text})


def escape_special_characters(s: str) -> str:
    """Returns a visually identifiable string for special characters."""
    return s.replace("\n", "\\n").replace("\t", "\\t")


def extract_error_snippet(
    error: json.JSONDecodeError, window: int = 20
) -> str:
    start = max(
        error.pos - window, 0
    )  # Start a bit before the error, if possible
    end = min(
        error.pos + window, len(error.doc)
    )  # End a bit after the error, if possible

    # Extract the snippet around the error
    snippet_start = error.doc[start : error.pos]
    snippet_error = error.doc[
        error.pos : error.pos + 1
    ]  # The erroneous character
    snippet_end = error.doc[error.pos + 1 : end]

    # Escape special characters in the erroneous part
    snippet_error_escaped = escape_special_characters(snippet_error)

    snippet = f"...{snippet_start}{RED}{snippet_error_escaped}{RESET}{snippet_end}..."
    return snippet


def trim_to_last_complete_sequence(
    json_string: str, end_sequences: List[str] = ["},", "}"]
) -> str:
    """
    Trims the given string from the end until it encounters any of the specified end sequences,
    indicating a point beyond which the string should be trimmed.

    Parameters:
    - json_string: The string to trim.
    - end_sequences: A list of strings indicating the sequences to look for as potential trim points.

    Returns:
    - A trimmed string ending with one of the complete sequences.
    """
    # Initialize variable to keep track of the earliest trim position found
    earliest_trim_position = len(json_string)

    # Iterate over each end sequence to find the last occurrence of each
    for sequence in end_sequences:
        sequence_position = json_string.rfind(sequence)
        if sequence_position != -1:
            # Update the trim position if this sequence occurs later than previous ones found
            trim_position = sequence_position + len(
                sequence
            )  # Include the sequence itself in the trimmed output
            earliest_trim_position = min(earliest_trim_position, trim_position)

    # If none of the sequences were found, return the original string
    if earliest_trim_position == len(json_string):
        return json_string

    # Return the substring up to and including the last occurrence of one of the sequences
    return json_string[:earliest_trim_position]


def merge_responses(previous_response: str, next_response: str) -> str:
    """
    Merges the next JSON response with the previous one, ensuring valid JSON structure.
    Specifically, it handles cases where individual JSON objects need to be part of a larger array.

    Parameters:
    - previous_response: The accumulated JSON response so far.
    - next_response: The latest JSON response fragment to merge.

    Returns:
    - A string representing the merged JSON response.
    """
    # Trim trailing whitespace to accurately check for closing characters
    trimmed_previous_response = previous_response.rstrip()
    trimmed_next_response = next_response.lstrip()

    # Check if we need to insert a comma to separate JSON objects
    if trimmed_previous_response.endswith(
        "}"
    ) and trimmed_next_response.startswith("{"):
        return trimmed_previous_response + "," + trimmed_next_response
    else:
        return trimmed_previous_response + trimmed_next_response


async def ask_llm(
    prompt: str,
    text: str,
    batch_number: int,
    total_batches: int,
    model_name: str,
    output_parser: Callable[[str], List[str]],
    fallback_model_name: Optional[str] = None,
    is_json: bool = True,  # Indicates if the response should be JSON
    extra_model_params: Optional[dict] = None,
) -> List[str]:
    client = get_openai_client(model_name)
    iteration = 0  # Initialize iteration counter
    incomplete_json = False  # Flag to indicate if the previous attempt failed due to incomplete JSON
    response = ""

    # Default model parameters
    default_model_params = {
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "frequency_penalty": DEFAULT_FREQUENCY_PENALTY,
    }

    # If extra_model_params is provided, update the default_model_params with it
    if extra_model_params is not None:
        default_model_params.update(extra_model_params)

    logging.info(
        f"[{model_name}] default_model_params : {default_model_params}"
    )

    while iteration < MAX_RETRIES:
        try:
            logging.info(
                f"[{model_name}] Sending request for batch {batch_number}/{total_batches}: {text}"
            )

            # TODO: pass in model params, else use default
            model_params = {
                **default_model_params,  # Spread the default (or updated) model parameters
                "model": model_name,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text},
                ],
            }

            if model_name in OPENAI_JSON_MODE_SUPPORTED_MODELS:
                model_params["response_format"] = {"type": "json_object"}

            if iteration == 0 or not incomplete_json:
                # Initial request or a retry not caused by incomplete JSON
                model_params["messages"] = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text},
                ]
            else:
                # Retry due to incomplete JSON, include continuation prompt and partial response

                model_params["messages"] = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text},
                    {
                        "role": "assistant",
                        "content": response,
                    },  # Include the last partial JSON response
                    {"role": "user", "content": CONTINUE_PROMPT},
                ]

                # del model_params["response_format"]next_response

                # logging.info(
                #     f"[{model_name}] {BLUE}Retrying request for batch due to incomplete JSON {batch_number}/{total_batches}: {model_params}{RESET}"
                # )

            # logging.info(
            #     f"[{model_name}] {BLUE}Sending request for batch {batch_number}/{total_batches}: {model_params}{RESET}"
            # )

            completion = await client.chat.completions.create(**model_params)
            next_response = completion.choices[0].message.content
            response = merge_responses(response, next_response)

            logging.info(
                f"[{model_name}] Received next response for batch {batch_number}/{total_batches}: {response}"
            )

            # TODO: debug special character
            logging.info(
                f"[{model_name}] {YELLOW}Merged response for batch {batch_number}/{total_batches}: {response}{RESET}"
            )

            # Reset the incomplete_json flag for the next iteration
            incomplete_json = False

            if is_json:
                # Attempt to parse the JSON to check completeness
                json.loads(response)
                logging.info(
                    f"[{model_name}] Received complete JSON response for batch {batch_number}/{total_batches}"
                )

            # If successful, process and return the parsed output
            parsed_output = output_parser(response)
            return parsed_output

        except json.JSONDecodeError as e:
            if not is_json:
                raise e  # If not expecting JSON, re-raise the exception

            # Set the flag indicating that this retry will be due to incomplete JSON
            incomplete_json = True
            logging.warning(
                f"[{model_name}] Received incomplete JSON, attempting to repair and continue."
            )

            response = trim_to_last_complete_sequence(
                response,
                #   end_sequences=["},", "}"]
                end_sequences=['"student_sentence_feedback": ['],
            )

            logging.info(f"[{model_name}] Repaired JSON: {response}")

        except Exception as e:
            logging.error(
                f"[{model_name}] An error occurred while processing: {e}"
            )

        # Increment the iteration after handling all exceptions
        iteration += 1
        if iteration >= MAX_RETRIES:
            return await handle_max_retries(
                model_name,
                fallback_model_name,
                prompt,
                text,
                batch_number,
                total_batches,
                output_parser,
                is_json,
            )

    # If loop exits due to reaching MAX_RETRIES
    logging.error(
        f"[{model_name}] Failed to complete JSON or recover from error after maximum iterations."
    )
    raise RuntimeError("Maximum iteration limit reached.")


async def handle_max_retries(
    model_name,
    fallback_model_name,
    prompt,
    text,
    batch_number,
    total_batches,
    output_parser,
    is_json,
):
    if fallback_model_name:
        logging.info(
            f"[{model_name}] Max retries reached, switching to fallback model: {fallback_model_name}"
        )
        # Update the client to use the fallback model
        # new_client = get_openai_client(fallback_model_name)
        # Retry the request with the fallback model, starting the iteration process again but keeping the batch context
        try:
            fallback_response = await ask_llm(
                prompt=prompt,
                text=text,
                batch_number=batch_number,
                total_batches=total_batches,
                model_name=fallback_model_name,
                output_parser=output_parser,
                fallback_model_name=None,  # Ensure no further fallback attempts
                is_json=is_json,
            )
            return fallback_response
        except Exception as e:
            # If the fallback attempt also fails, log the error and exit
            logging.error(
                f"[{fallback_model_name}] Fallback attempt failed: {e}"
            )
            sys.exit(1)
    else:
        # No fallback model specified, or fallback attempt failed
        logging.error(
            f"[{model_name}] Max retries reached with no available fallback. Exiting."
        )
        sys.exit(1)


async def mock_gec_system(
    input_sentences: List[str],
    model_id: str,
    fallback_model_id: Optional[str] = None,
) -> tuple:
    # Simulate processing of input sentences by a mock GEC system
    # Utilize ModelIOParser for preparing input and parsing output
    prepared_input = ModelIOParser.prepare_model_input(input_sentences)
    # TODO: Implement the mock GEC system logic here

    model_output = await ask_llm(
        prompt=GRAMMAR_PROMPT,
        text=prepared_input,
        batch_number=1,
        total_batches=1,
        model_name=model_id,
        fallback_model_name=fallback_model_id,
        output_parser=lambda response: ModelIOParser.parse_model_output(
            response, input_sentences
        ),
    )

    return model_id, model_output


# Additional components (Aggregate Node, Condition Node, etc.) remain similar
# to the previous code skeleton and should be implemented accordingly
async def extract_edits(aggregated_responses, input_sentences):
    """
    Extracts edits from corrected sentences using ERRANT.

    :param aggregated_responses: Dictionary with model IDs as keys and lists of corrected sentences as values.
    :param input_sentences: List of original sentences.
    :return: Dictionary with model IDs as keys and lists of edits for each sentence as values.
    """
    edits_output = {}

    for model_id, corrected_sentences in aggregated_responses.items():
        model_edits = []
        for original_sentence, corrected_sentence in zip(
            input_sentences, corrected_sentences
        ):
            # Parse the original and corrected sentences
            orig_doc = nlp(original_sentence)
            cor_doc = nlp(corrected_sentence)

            # Generate ERRANT edits
            edits = annotator.annotate(orig_doc, cor_doc)

            # Convert edits to M2 format
            edit_list = [edit.to_m2() for edit in edits]

            sentence_output = {
                "original_sentence": original_sentence,
                "corrected_sentence": corrected_sentence,
                "edits": edit_list,
            }

            model_edits.append(sentence_output)

        edits_output[model_id] = model_edits

    return edits_output


def calculate_edit_votes(edits_output):
    """
    Calculates votes for each edit operation across all models.

    :param edits_output: Dictionary with model IDs as keys and lists of edits for each sentence as values.
    :return: Dictionary with edit operations as keys and votes (counts) as values.
    """
    edit_votes = {}

    for model_edits in edits_output.values():
        for sentence_edits in model_edits:
            for edit in sentence_edits["edits"]:
                # Assuming the edit format is consistent and can be used as a dictionary key
                if edit not in edit_votes:
                    edit_votes[edit] = 1
                else:
                    edit_votes[edit] += 1

    return edit_votes


async def quality_estimation_node(
    input_sentences: List[str],
    aggregated_responses: Dict[str, List[str]],
    model_ids: List[str],
    quality_estimation_model_id: str,
):
    """
    Sends all original sentences vs. corrected sentences in one prompt to the LLM for quality estimation.

    :param aggregated_responses: Dictionary with model IDs as keys and lists of corrected sentences as values.
    :param model_ids: List of model identifiers.
    :param client: The client used to communicate with the LLM.
    :return: A dictionary mapping model names to quality scores for each sentence.
    """
    quality_scores = {}

    # Logging information about the function start
    logging.info("Starting quality estimation node.")

    for model_id in model_ids:

        corrected_sentences = aggregated_responses[model_id]

        # Construct JSON input for the prompt
        text = {
            "original_sentence": input_sentences,
            "student_sentence": corrected_sentences,
        }

        prompt = QUALITY_ESTIMATION_PROMPT_V2

        def output_parser(response: str, expected_num_sentences: int):
            try:
                data = json.loads(response)
                evaluations = data.get("evaluations", [])
                if len(evaluations) != expected_num_sentences:
                    raise ValueError(
                        f"Expected {expected_num_sentences} evaluations, but got {len(evaluations)}."
                    )

                # Extract the scores from each evaluation
                scores = [evaluation["score"] for evaluation in evaluations]
                return scores
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to decode JSON response: {str(e)}")

        expected_num_sentences = len(corrected_sentences)

        # Logging information about the model being processed
        logging.info(f"Processing model: {model_id}")

        extra_model_params = {
            "frequency_penalty": QUALITY_ESTIMATION_FREQUENCY_PENALTY,
        }

        # Making an assumption about the ask_llm function call; adapt as necessary
        quality_scores[model_id] = await ask_llm(
            prompt=prompt,  # Now includes structured instructions for processing JSON input
            text=json.dumps(text),  # Passes the constructed JSON as input
            batch_number=1,
            total_batches=1,
            model_name=quality_estimation_model_id,
            output_parser=lambda response: output_parser(
                response, expected_num_sentences
            ),
            extra_model_params=extra_model_params,
        )

    # Logging information about the function completion
    logging.info("Quality estimation node completed.")

    return quality_scores


# TODO: fix this node


async def quality_adjustment_node(
    quality_scores: Dict[str, List[float]],
    edit_votes: Dict[str, int],
    edits_output: Dict[str, List[Dict[str, Any]]],
):
    """
    Adjusts the quality scores based on the edit votes, allowing scores to exceed 100 for tie-breaking.
    """
    adjusted_quality_scores = {}

    for model_id, scores in quality_scores.items():
        adjusted_scores = []
        for score, sentence_edits in zip(scores, edits_output[model_id]):
            # Ensure the initial score is within the correct range
            initial_score = max(0, min(score, 100))

            # Calculate total votes for the edits of the current sentence
            total_votes = sum(
                edit_votes.get(edit, 0) for edit in sentence_edits["edits"]
            )

            # Calculate total adjustment based on the vote increase factor
            total_adjustment = total_votes * VOTE_INCREASE_FACTOR

            # Adjust the quality score
            adjusted_score = initial_score + total_adjustment

            # Allow scores to exceed 100 but cap at MAX_SCORE_CAP
            adjusted_score = min(adjusted_score, MAX_SCORE_CAP)
            adjusted_scores.append(adjusted_score)

        adjusted_quality_scores[model_id] = adjusted_scores

    return adjusted_quality_scores


async def system_combination_node(
    adjusted_quality_scores: Dict[str, List[float]],
    aggregated_responses: Dict[str, List[str]],
):
    """
    Selects the best sentences based on the majority voting from adjusted quality scores.

    :param adjusted_quality_scores: Dictionary mapping model names to adjusted quality scores for each sentence.
    :param aggregated_responses: Dictionary with model IDs as keys and lists of corrected sentences as values.
    :return: List of the best sentences chosen among the models.
    """
    best_sentences = []
    total_scores = {
        model_id: sum(scores)
        for model_id, scores in adjusted_quality_scores.items()
    }

    # Determine the number of sentences
    num_sentences = len(next(iter(aggregated_responses.values())))

    for i in range(num_sentences):
        # Collect scores for this sentence across all models
        sentence_scores = {
            model_id: scores[i]
            for model_id, scores in adjusted_quality_scores.items()
        }

        # Determine the highest score for the current sentence
        max_score = max(sentence_scores.values())

        # Get models with the highest score for this sentence
        top_models = [
            model_id
            for model_id, score in sentence_scores.items()
            if score == max_score
        ]

        # If multiple models have the top score, use the total adjusted quality score as a tiebreaker
        if len(top_models) > 1:
            best_model = max(
                top_models, key=lambda model_id: total_scores[model_id]
            )
        else:
            best_model = top_models[0]

        # Select the sentence from the best model
        best_sentences.append(aggregated_responses[best_model][i])

    return best_sentences


async def execute_workflow(input_string: str):
    input_sentences = InputParser.parse_input(input_string)
    model_ids = [
        OPENAI_JSON_MODE_SUPPORTED_MODELS[0]
        #  , TOGETHER_AI_MODELS[1]
    ]

    # Asynchronously call mock_gec_system for each model_id
    tasks = [
        mock_gec_system(input_sentences, model_id) for model_id in model_ids
    ]
    model_responses = await asyncio.gather(*tasks)

    # Aggregate model responses
    aggregated_responses = {
        model_id: response for model_id, response in model_responses
    }

    # Initiate Quality Estimation and Edit Extraction concurrently
    quality_estimation_task = quality_estimation_node(
        input_sentences,
        aggregated_responses,
        model_ids,
        QUALITY_ESTIMATION_MODEL_NAME,
    )
    edit_extraction_task = extract_edits(
        aggregated_responses, input_sentences
    )  # Adjust this if extract_edits is async

    # Wait for both tasks to complete
    quality_estimation, edits_output = await asyncio.gather(
        quality_estimation_task, edit_extraction_task
    )

    # Calculate edit votes after edit extraction is done
    edit_votes = calculate_edit_votes(edits_output)

    # Adjust quality scores based on edit votes
    adjusted_quality_scores = await quality_adjustment_node(
        quality_estimation, edit_votes, edits_output
    )

    # System combination to select the best sentences
    best_sentences = await system_combination_node(
        adjusted_quality_scores, aggregated_responses
    )

    # Now, after both quality estimation, edit votes calculation, and system combination are done, print out the results
    logging.info("Edit Extraction:")
    logging.info(json.dumps(edits_output, indent=2))

    logging.info("Voting Bias:")
    logging.info(json.dumps(edit_votes, indent=2))

    logging.info("Quality Estimation:")
    logging.info(json.dumps(quality_estimation, indent=2))

    logging.info("Adjusted Quality Scores:")
    logging.info(json.dumps(adjusted_quality_scores, indent=2))

    logging.info("Best Sentences Selected:")
    logging.info(json.dumps(best_sentences, indent=2))

    # After all processing is done, create the final output
    final_output = "\n".join(best_sentences)

    # Return the final output instead of printing or logging
    return final_output


async def read_input_file(file_path: str) -> str:
    """Reads the input file and returns its content."""
    async with aiofiles.open(file_path, "r") as f:
        return await f.read()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument(
        "input_text",
        nargs="?",
        default=None,
        help="The input text to process. Optional.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Run in quiet mode, producing only the final output.",
    )
    args = parser.parse_args()

    # print("input_text", args.quiet)
    # print("--quiet", args.quiet)

    # Determine the input_string based on args.input_text
    if args.input_text:
        input_string = args.input_text
    else:
        # If no input text is provided, you might read from a default file or another source
        input_string = asyncio.run(read_input_file(TEST_FILE_PATH))

    # Execute the workflow
    output = asyncio.run(execute_workflow(input_string))

    print(output)
