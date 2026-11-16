from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

MODEL_PATH = Path("models/career_model.joblib")

SKILLS_BY_ROLE = {
    "Data Scientist": ["python", "statistics", "machine learning", "pandas", "sql", "data visualization"],
    "AI/ML Engineer": ["python", "machine learning", "deep learning", "scikit-learn", "tensorflow", "pytorch"],
    "Web Developer": ["html", "css", "javascript", "react", "api", "git"],
    "Cybersecurity Analyst": ["networking", "linux", "security", "siem", "risk assessment", "python"],
    "Cloud Engineer": ["aws", "azure", "cloud", "docker", "kubernetes", "linux"],
}

def train_model(csv_path="data/job_roles.csv", model_path=MODEL_PATH):
    data = pd.read_csv(csv_path).dropna(subset=["description", "role"])
    if data["role"].nunique() < 2 or len(data) < 10:
        raise ValueError("Training data needs at least 10 rows and 2 distinct roles.")
    X_train, X_test, y_train, y_test = train_test_split(
        data["description"].astype(str), data["role"].astype(str),
        test_size=0.25, random_state=42, stratify=data["role"]
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=5000)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "report": classification_report(y_test, predictions, zero_division=0),
        "test_rows": len(y_test),
    }

def load_model(model_path=MODEL_PATH):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError("Model not found. Run: python train.py")
    return joblib.load(path)

def recommend_roles(text, model, top_n=3):
    probabilities = model.predict_proba([text])[0]
    classes = model.classes_
    ranked = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
    return [{"role": role, "score": float(score)} for role, score in ranked[:top_n]]

def extract_skills(text):
    lowered = text.lower()
    vocabulary = sorted({skill for skills in SKILLS_BY_ROLE.values() for skill in skills})
    return [skill for skill in vocabulary if skill in lowered]

def skill_gap(text, role):
    present = set(extract_skills(text))
    required = SKILLS_BY_ROLE.get(role, [])
    return {
        "matched": [skill for skill in required if skill in present],
        "missing": [skill for skill in required if skill not in present],
    }
