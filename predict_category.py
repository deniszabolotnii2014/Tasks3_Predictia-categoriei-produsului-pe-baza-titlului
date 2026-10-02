from pathlib import Path
import pickle

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Nu găsesc modelul:\n{MODEL_PATH}\n\n"
            "Rulează mai întâi: python train_model.py"
        )

    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def predict_title(title, artifact):
    model = artifact["model"]
    word_vectorizer = artifact["word_vectorizer"]
    char_vectorizer = artifact["char_vectorizer"]

    X_word = word_vectorizer.transform([title])
    X_char = char_vectorizer.transform([title])

    from scipy.sparse import hstack

    X = hstack([X_word, X_char])

    prediction = model.predict(X)[0]

    return prediction

def main():
    print("=" * 60)
    print("PRODUCT CATEGORY PREDICTOR")
    print("=" * 60)

    print(f"Loading model from: {MODEL_PATH}")

    artifact = load_model()

    print("Model loaded successfully!")
    print()
    print("Introdu un titlu de produs.")
    print("Scrie 'exit' pentru a ieși.")
    print()

    while True:
        title = input("Product title: ").strip()

        if title.lower() in {"exit", "quit"}:
            print("Program terminated.")
            break

        if not title:
            print("Please enter a non-empty title.")
            continue

        category = predict_title(title, artifact)

        print(f"Predicted category: {category}")
        print()

if __name__ == "__main__":
    main()