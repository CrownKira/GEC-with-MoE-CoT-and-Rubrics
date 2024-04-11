import sys
import json
import asyncio
from typing import List, Dict, Union, Tuple, Any
from dotenv import load_dotenv
import spacy
import os
import groq
import openai
from clients.coze import AsyncCoze


# CONFIGS: API
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")
LOCAL_ENDPOINT = os.getenv("LOCAL_ENDPOINT", "")
TOGETHER_ENDPOINT = os.getenv("TOGETHER_ENDPOINT", "")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COZE_API_KEY = os.getenv("COZE_API_KEY", "")
MAX_RETRIES = 3  # Maximum number of retries for an API call
RETRY_DELAY = 5  # Delay in seconds before retrying an API
QPM_LIMIT = 5  # Queries per minute limit


# CONFIGS: PROMPT
# GRAMMAR_VARIANT = "standard American"
GRAMMAR_VARIANT = "British"
TEXT_DELIMITER = "~~~"

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


# change model here
MODEL_NAME = OPENAI_JSON_MODE_SUPPORTED_MODELS[0]


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


client = get_openai_client(MODEL_NAME)


# Load the spaCy model outside of the asynchronous function to avoid reloading it multiple times
nlp = spacy.load("en_core_web_sm")

# Load environment variables from .env file
load_dotenv()


# Configuration variable for model injection
MODEL_CONFIG = {
    "Model_1": {"input_parser": None, "model": None, "output_parser": None},
    "Model_2": {"input_parser": None, "model": None, "output_parser": None},
    # Add more models as needed
}

# Define the Input Parser for command line input
def parse_input_from_command_line() -> List[str]:
    input_text = sys.argv[
        1
    ]  # Assuming the input is provided as the first command line argument
    input_sentences = input_text.split("\n")
    return input_sentences


# Define the Input Parser
def input_parser(sentences: List[str]) -> str:
    # Join the sentences with the delimiter and return as JSON
    return json.dumps({"text": TEXT_DELIMITER.join(sentences)})


# Define the Model Node (Mock GEC System)
async def model_node(input_json: str) -> str:
    # TODO: Implement the mock GEC system logic for the given model
    # For now, we just return the input as the output for demonstration purposes
    return input_json


# Define the Output Parser
def output_parser(model_output_json: str) -> Dict[str, Any]:
    try:
        # Deserialize the JSON string to get the actual text
        text_data = json.loads(model_output_json)
        input_text = text_data["text"]
        # Splitting the deserialized text into lines based on the custom delimiter
        sentences = input_text.split(TEXT_DELIMITER)
        return {"sentences": sentences, "error": ""}
    except (json.JSONDecodeError, KeyError) as e:
        return {"sentences": [], "error": f"Processing failed due to {str(e)}"}


# Define the Aggregate Node
def aggregate_node(
    model_responses: List[Dict[str, Union[List[str], str]]]
) -> Dict[str, Dict[str, Union[List[str], str]]]:
    # TODO: Implement the logic to aggregate responses from all model nodes
    pass


# Define the Condition Node
def condition_node(
    aggregated_responses: Dict[str, Dict[str, Union[List[str], str]]]
) -> bool:
    # TODO: Implement the logic to check for errors in the aggregated responses
    pass


# Define the Quality Estimation Node
def quality_estimation_node(
    sentences_for_quality_check: List[str],
) -> Dict[str, List[Tuple[str, float]]]:
    # TODO: Implement the logic to estimate quality of corrected sentences
    pass


# Define the Edit Extraction Node
def edit_extraction_node(
    aggregated_responses: Dict[str, Dict[str, Union[List[str], str]]]
) -> Dict[str, List[Dict[str, Union[str, List[str]]]]]:
    # TODO: Implement the logic to extract edits from the corrected sentences
    pass


# Define the Voting Bias Node
def voting_bias_node(
    edits_list: List[Dict[str, Union[str, List[str]]]]
) -> List[Tuple[str, int]]:
    # TODO: Implement the logic to calculate voting scores for edits
    pass


# Define the Quality Adjustment Node
def quality_adjustment_node(
    quality_scores_with_edits: Dict[str, List[Tuple[str, float, List[str]]]]
) -> Dict[str, List[Tuple[str, float]]]:
    # TODO: Implement the logic to adjust quality scores based on edits and their voting scores
    pass


# Define the System Combination Node
def system_combination_node(
    candidates_for_final_selection: Dict[str, List[Tuple[str, float]]]
) -> List[str]:
    # TODO: Implement the logic to combine system outputs into final corrections
    pass


# Define the Output Node
def output_node(final_output: List[str]) -> List[str]:
    # TODO: Implement the logic to output the final corrected sentences or error message
    pass


# Define the Input Parser Node
def input_parser_node(input_data: str) -> List[str]:
    # TODO: Implement the logic to parse input sentences
    pass


# Define the Output Parser Node
def output_parser_node(
    model_response: Dict[str, Union[List[str], str]]
) -> Union[List[str], None]:
    # TODO: Implement the logic to parse the corrected sentences from the model response
    pass


# Define the Retry Node
def retry_node(current_retry_count: int, max_retries: int) -> bool:
    # TODO: Implement the logic to determine if a retry should be attempted
    pass


# Define the Variable Node for Max Retry Count
def get_max_retry_count() -> int:
    # TODO: Define the maximum number of retries allowed
    pass


# Define the Condition Node for Retry Logic
def should_retry(current_retry_count: int, max_retries: int) -> bool:
    # TODO: Implement the logic to check if the retry count has been exceeded
    pass


# Define the End Node for Corrected Sentences
def end_node_corrected_sentences(corrected_sentences: List[str]) -> List[str]:
    # TODO: Implement the logic to handle the end of the process with corrected sentences
    pass


# Define the End Node for Error Message
def end_node_error_message(error_message: str) -> str:
    # TODO: Implement the logic to handle the end of the process with an error message
    pass


# Main function to run the workflow
async def main():
    input_sentences = parse_input_from_command_line()
    aggregated_responses = {}

    # Process each model
    for model_name, model_info in MODEL_CONFIG.items():
        # Parse input for the model
        model_input = input_parser_for_model("\n".join(input_sentences))

        # Call the model node (asynchronously)
        model_output = await model_node(input_sentences, model_name)

        # Parse output from the model
        parsed_output = output_parser_for_model(model_output)

        # Aggregate responses
        aggregated_responses[model_name] = parsed_output

    # TODO: Continue with the rest of the workflow using the aggregated responses


# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
