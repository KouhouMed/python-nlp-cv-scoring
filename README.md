# Python NLP CV Scoring

This project implements a Natural Language Processing (NLP) based system for scoring and analyzing Curriculum Vitae (CV) or resumes.

## Features

- CV text extraction from various file formats (PDF, DOCX, etc.)
- NLP-based analysis of CV content
- Scoring system based on predefined criteria
- Customizable scoring parameters
- Generation of detailed reports

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/KouhouMed/python-nlp-cv-scoring
   ```

2. Navigate to the project directory:
   ```
   cd python-nlp-cv-scoring
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Place CV files in the `data/cvs` directory and job description files in the `data/job_descriptions` directory.
2. Run the main script:
   ```
   python main.py
   ```
3. Find the generated reports in the `output` directory.

## Configuration

Adjust the scoring parameters in `config.yaml` to customize the CV evaluation criteria.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
