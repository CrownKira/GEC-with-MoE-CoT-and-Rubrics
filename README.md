# GEC-CoT-Rubrics

## GEC System Architecture
![Untitled Diagram7-Page-1 drawio](https://github.com/CrownKira/GEC-with-MoE-CoT-and-Rubrics
/assets/24221801/a28c9e53-4857-4c10-8954-b0c49e8893c4)

Our system leverages NLP techniques to address a wide range of grammatical errors, providing high-quality corrections. By integrating Chain-of-Thought (CoT) prompting with rubric-guided evaluations, our models achieve significant performance improvements in quality estimation tasks.

## Overview

Our GEC system architecture employs a novel combination of technologies and methodologies to enhance the accuracy and diversity of grammatical error corrections:

- **Mixture of Experts**: Utilizes a specialized approach where each "student" model is an expert on one of four datasets—A, B, C, or N—achieved through fine-tuning on each specific dataset.
- **Chain-of-Thought Prompting**: Utilizes structured reasoning prompts to guide the model towards more logical and contextually relevant corrections.
- **Rubric-Guided Evaluations**: Incorporates specific evaluation criteria, ensuring that corrections adhere to predefined quality standards.
- **In-Context Learning**: Leverages the capabilities of models with advanced reasoning abilities, notably GPT-3.5-turbo-1106, for dynamic learning within contextual boundaries.
- **Teacher and Student Model Integration**: Combines the insights of teacher models in evaluating student corrections, fostering a rich pool of diverse and insightful corrections.

## Installation

### Steps

1. **Clone the Repository:**

   ```bash
   git clone git@github.com:CrownKira/GEC-with-MoE-CoT-and-Rubrics
.git
   ```

2. **Navigate to the Project Directory:**

   ```bash
   cd GEC-with-MoE-CoT-and-Rubrics

   ```

3. **Create and Activate the Virtual Environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install Required Python Packages:**

   ```bash
   pip3 install -r requirements.txt
   ```

5. **Download the Necessary spaCy Language Model:**
   ```bash
   python3 -m spacy download en_core_web_sm
   ```

## Usage

1. **Configure Environment Variables:**
   Obtain the `.env` file from me or set your own environment variables as needed.

2. **Run the Main Script:**
   Replace `main.py` with the actual name of your script.

   ```bash
   python3 main.py
   ```

3. **Check Outputs:**
   Navigate to the `outputs` directory to access corrected text files and CSV outputs.
