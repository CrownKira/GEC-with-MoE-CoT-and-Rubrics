import spacy
from spacy.tokens import Doc
from typing import List


def tokenize_text(corrected_text: str) -> str:
    nlp = spacy.load("en_core_web_sm")
    doc: Doc = nlp(corrected_text.strip())
    processed_text: str = " ".join(token.text for token in doc)
    stripped_lines: List[str] = [
        line.strip() for line in processed_text.split("\n")
    ]
    processed_text: str = "\n".join(stripped_lines)
    return processed_text


def main():
    # text = "This is a sample text to be tokenized."
    text = """{
    "total_sentences": 4,
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
}"""
    tokenized_text = tokenize_text(text)
    print(tokenized_text)


if __name__ == "__main__":
    main()
