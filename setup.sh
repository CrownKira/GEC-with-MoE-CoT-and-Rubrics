#!/bin/bash

# Install Python packages
pip3 install -r requirements.txt

# Download the spaCy language model
python3 -m spacy download en_core_web_sm
