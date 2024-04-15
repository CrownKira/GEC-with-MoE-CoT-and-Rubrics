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
    #     text = """{
    #     "total_sentences": 4,
    #     "evaluations": [
    #         {
    #             "unique_index": 0,
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
    #             "unique_index": 1,
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
    #             "unique_index": 2,
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
    #             "unique_index": 3,
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
    # }"""

    text = """{
    "total_sentences": 4,
    "evaluations": [
        {
            "unique_index": 0,
            "student_sentence": "Has you told me that I will win some literary competitions and that some people will speak well of me , I would n't have believe you .",
            "student_sentence_feedback": [
                {
                    "type": "GRAM-MAJ",
                    "description": "The phrase 'Has you told me' should be corrected to 'Had you told me' to correctly form the conditional perfect tense.",
                    "deduction": -7
                },
                {
                    "type": "GRAM-MAJ",
                    "description": "Missing 'would' before 'win' to maintain conditional tense consistency: 'I would win some literary competitions.'",
                    "deduction": -7
                },
                {
                    "type": "GRAM-MAJ",
                    "description": "'believe' should be in the past participle form as 'believed' to correctly use the conditional perfect tense.",
                    "deduction": -7
                }
            ],
            "corrected_sentence": "Had you told me that I would win some literary competitions and that some people would speak well of me, I wouldn't have believed you.",
            "total_deductions": -21,
            "score": 79
        },
        {
            "unique_index": 1,
            "student_sentence": "The year was 2012 and I had n't written anything until that day - I just had been translating some stories and once even subtitles for a Korean movie from English and Spanish .",
            "student_sentence_feedback": [
                {
                    "type": "PUNCT-MIN",
                    "description": "A comma should be added after '2012' to correctly separate clauses.",
                    "deduction": -2
                },
                {
                    "type": "GRAM-MIN",
                    "description": "The phrase 'had just been' should be reordered to 'had been just' for correct word order.",
                    "deduction": -3
                },
                {
                    "type": "PUNCT-MIN",
                    "description": "A comma should be placed before 'and once even subtitles' for clarity and proper separation of clauses.",
                    "deduction": -2
                },
                {
                    "type": "PUNCT-MIN",
                    "description": "A comma should be added after 'subtitles for a Korean movie' for correct clause separation.",
                    "deduction": -2
                }
            ],
            "corrected_sentence": "The year was 2012, and I hadn't written anything until that day - I had been just translating some stories, and once, even subtitles for a Korean movie, from English and Spanish.",
            "total_deductions": -9,
            "score": 91
        },
        {
            "unique_index": 2,
            "student_sentence": "But that day - it was on spring and I believe it was Thursday - my English teacher told us about a literary competition .",
            "student_sentence_feedback": [
                {
                    "type": "LANG-FORM",
                    "description": "'on spring' should be corrected to 'in spring' to use the correct preposition for seasons.",
                    "deduction": -6
                }
            ],
            "corrected_sentence": "But that day - it was in spring and I believe it was Thursday - my English teacher told us about a literary competition.",
            "total_deductions": -6,
            "score": 94
        }
    ]
}
"""

    tokenized_text = tokenize_text(text)
    print(tokenized_text)


if __name__ == "__main__":
    main()
