from pathlib import Path
import pickle
import pandas as pd

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Folderul în care se află train_model.py
BASE_DIR = Path(__file__).resolve().parent

# Fișierele proiectului
DATA_PATH = BASE_DIR / "IMLP4_TASK_03-products.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

CATEGORY_MAPPING = {
    "fridge": "Fridge Freezers",
    "CPU": "CPUs",
    "Mobile Phone": "Mobile Phones",
}

def clean_data(df):
    df = df.copy()

    # Eliminam spatiile accidentale din numele coloanelor
    df.columns = df.columns.str.strip()

    df["Product Title"] = (
        df["Product Title"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["Category Label"] = (
        df["Category Label"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["Category Label"] = df["Category Label"].replace(CATEGORY_MAPPING)

    # Eliminăm produsele fără titlu sau categorie
    df = df[
        (df["Product Title"] != "") &
        (df["Category Label"] != "")
    ]   

    return df

def main():
    print("Loading dataset...")
    print(f"Dataset: {DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Nu găsesc datasetul:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset loaded: {df.shape}")

    df = clean_data(df)

    X = df["Product Title"]
    y = df["Category Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Word-level TF-IDF
    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=100000,
        sublinear_tf=True
    )

    # Character-level TF-IDF
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100000,
        sublinear_tf=True
    )

    print("Training TF-IDF vectorizers...")

    X_train_word = word_vectorizer.fit_transform(X_train)
    X_test_word = word_vectorizer.transform(X_test)

    X_train_char = char_vectorizer.fit_transform(X_train)
    X_test_char = char_vectorizer.transform(X_test)

    X_train_final = hstack([X_train_word, X_train_char])
    X_test_final = hstack([X_test_word, X_test_char])

    print("Training LinearSVC...")

    model = LinearSVC(
        C=2.0,
        random_state=42
    )

    model.fit(X_train_final, y_train)
    predictions = model.predict(X_test_final)
    accuracy = accuracy_score(y_test, predictions)

    print()
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f}")
    print("=" * 60)

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # Salvăm toate componentele necesare predicției
    artifact = {
        "model": model,
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "category_mapping": CATEGORY_MAPPING,
        "accuracy": accuracy
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)

    print()
    print(f"Model saved successfully:")
    print(MODEL_PATH)

if __name__ == "__main__":
    main()