# 🧭 CareerCompass AI — Resume Skill Gap & Career Recommender

CareerCompass AI is an educational AI/ML portfolio project. It analyses resume text, predicts related career domains using TF-IDF + Logistic Regression, and compares detected skills with a small curated skill checklist.

## Features
- Paste resume text or upload a text-based PDF
- TF-IDF text feature extraction
- Logistic Regression multi-class classifier
- Top-three career-domain recommendations
- Skill-match and learning-gap checklist
- Model training script and unit tests
- Streamlit web interface

> **Limitations:** The included job-description dataset is small and illustrative. Model scores are not calibrated probabilities and do not measure a person's actual employability. Skill matching is keyword-based. Do not upload sensitive personal information to public demos.

## Project structure

```text
CareerCompass-AI/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── job_roles.csv
├── src/
│   ├── __init__.py
│   └── analyzer.py
└── tests/
    └── test_analyzer.py
```

## Run locally

Python 3.10+ recommended.

```bash
git clone https://github.com/YOUR-USERNAME/CareerCompass-AI.git
cd CareerCompass-AI

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Dataset format

`data/job_roles.csv` contains two columns: `description` and `role`. Replace or extend it with a properly licensed dataset of job descriptions, balanced across roles and cleaned for duplicates.

## Evaluation

`python train.py` prints hold-out accuracy and a classification report. The starter data is intentionally small, so its score is only a pipeline demonstration. For a serious study, gather a larger representative dataset, group near-duplicate descriptions before splitting, compare multiple models, and report precision, recall, F1-score, and confusion matrix.

## Future improvements
- Add resume section parsing and named-entity recognition
- Compare Linear SVM, Naive Bayes, and Logistic Regression
- Add semantic embeddings for better synonym matching
- Let users edit the skill requirements for each target role
- Add model cards, dataset licence details, and bias evaluation

## Tech stack
Python • Pandas • scikit-learn • TF-IDF • Logistic Regression • Streamlit • pypdf • pytest
