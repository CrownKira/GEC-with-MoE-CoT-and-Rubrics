import json


class InputParser:
    @staticmethod
    def parse_input(input_string):
        # Parse a string of \n separated sentences into a list
        return input_string.strip().split("\n")


class ModelIOParser:
    @staticmethod
    def parse_model_output(model_output, input_sentences):
        try:
            # Deserialize the JSON string to get the actual text
            text_data = json.loads(model_output)
            input_text = text_data["text"]
            # Splitting the deserialized text into lines based on the custom delimiter
            sentences = input_text.split("~~~")

            expected_num_sentences = len(input_sentences.split("\n"))
            actual_num_sentences = len(sentences)

            if actual_num_sentences < expected_num_sentences:
                # Calculate the difference in the number of lines
                diff = expected_num_sentences - actual_num_sentences
                return {
                    "sentences": [],
                    "error": f"Processing failed due to insufficient number of lines. Expected at least {expected_num_sentences}, but got {actual_num_sentences}. Difference: {diff}.",
                }

            # If successful, return the sentences with no error
            return {"sentences": sentences, "error": ""}

        except (json.JSONDecodeError, KeyError) as e:
            # Return an error message if parsing fails or the required fields are not found
            return {
                "sentences": [],
                "error": f"Processing failed due to {str(e)}",
            }

    @staticmethod
    def prepare_model_input(input_text):
        # Splitting the input text into lines, and then joining them with the custom delimiter
        joined_text = "~~~".join(input_text.split("\n"))
        # Wrapping the joined text in a dictionary and serializing it to a JSON string
        return json.dumps({"input": joined_text})


def mock_gec_system(input_sentences, model_id):
    # Simulate processing of input sentences by a mock GEC system
    # Utilize ModelIOParser for preparing input and parsing output
    prepared_input = ModelIOParser.prepare_model_input(input_sentences)
    # TODO: Implement the mock GEC system logic here
    model_output = {
        "text": "If I had to choose between both transportation, I think I would probably choose a car because it is better for me to go by car than to go by bus.~~~In my community, we are very interested in the environment and ecological things.~~~We have solar panels and a place to make compost in the last garden, with worms that eat and degrade all the organic waste of the school."
    }
    parsed_output = ModelIOParser.parse_model_output(
        model_output, input_sentences
    )
    return model_id, parsed_output


# Additional components (Aggregate Node, Condition Node, etc.) remain similar
# to the previous code skeleton and should be implemented accordingly


def execute_workflow(input_string):
    input_sentences = InputParser.parse_input(input_string)
    model_ids = [1, 2, "N"]  # Example identifiers for mock GEC systems
    model_responses = [
        mock_gec_system(input_sentences, model_id) for model_id in model_ids
    ]
    # TODO: Implement the rest of the workflow using the previously defined components

    # This is just a placeholder to show where the rest of the workflow would be implemented
    print(
        "Workflow executed with the following input sentences:",
        input_sentences,
    )


# Example of how a user would call this script with a string of \n separated sentences
if __name__ == "__main__":
    import sys

    input_string = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Ths is an eror sentence.\nAnothr mistke."
    )
    execute_workflow(input_string)
