"""Train only Ted's intent-classification model.

This focused script keeps the original project's preprocessing and neural
network architecture, but removes the unrelated interactive services from
``full_code.py`` (web search, weather, news, cricket, songs, timer and Covid
lookups). It reads ``intents.json`` and writes:

- ``words.pkl``
- ``classes.pkl``
- ``mymodel.h5``

Example:
    python train_model.py --epochs 200 --seed 42
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer


IGNORE_TOKENS = {"?", "!", ",", "'s"}


def ensure_nltk_data() -> None:
    """Download the two NLTK resources this project needs, if missing."""
    for resource, package in (
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=True)


def load_training_data(intents_path: Path):
    lemmatizer = WordNetLemmatizer()
    intents = json.loads(intents_path.read_text(encoding="utf-8"))

    words: list[str] = []
    classes: list[str] = []
    documents: list[tuple[list[str], str]] = []

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)
            documents.append((tokens, intent["tag"]))
            if intent["tag"] not in classes:
                classes.append(intent["tag"])

    words = sorted(
        {
            lemmatizer.lemmatize(word.lower())
            for word in words
            if word not in IGNORE_TOKENS
        }
    )
    classes = sorted(set(classes))

    train_x: list[list[int]] = []
    train_y: list[list[int]] = []

    for tokens, tag in documents:
        lemmatized = [lemmatizer.lemmatize(token.lower()) for token in tokens]
        bag = [1 if word in lemmatized else 0 for word in words]
        output = [0] * len(classes)
        output[classes.index(tag)] = 1
        train_x.append(bag)
        train_y.append(output)

    return intents, words, classes, np.array(train_x), np.array(train_y), lemmatizer


def build_model(input_size: int, output_size: int):
    # Imported here so --help and data checks do not require Keras.
    # Any Keras 3 backend works; this project was retrained with the JAX CPU
    # backend because it is much smaller to install than TensorFlow.
    import keras

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_size,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(output_size, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def clean_up(sentence: str, lemmatizer: WordNetLemmatizer) -> list[str]:
    return [
        lemmatizer.lemmatize(token.lower())
        for token in nltk.word_tokenize(sentence)
    ]


def create_bow(sentence: str, words: list[str], lemmatizer: WordNetLemmatizer):
    sentence_words = clean_up(sentence, lemmatizer)
    return np.array(
        [[1 if word in sentence_words else 0 for word in words]], dtype=float
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain Ted's intent model")
    base_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=base_dir,
        help="Folder containing intents.json (default: this script's folder)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Folder for words.pkl, classes.pkl and mymodel.h5 (default: data dir)",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or data_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_nltk_data()
    intents, words, classes, train_x, train_y, lemmatizer = load_training_data(
        data_dir / "intents.json"
    )

    with open(output_dir / "words.pkl", "wb") as file:
        pickle.dump(words, file)
    with open(output_dir / "classes.pkl", "wb") as file:
        pickle.dump(classes, file)

    random.seed(args.seed)
    np.random.seed(args.seed)

    import keras

    keras.utils.set_random_seed(args.seed)
    model = build_model(train_x.shape[1], train_y.shape[1])
    history = model.fit(
        train_x,
        train_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
    )
    model.save(output_dir / "mymodel.h5")

    # Reload exactly what was saved, then test the classifier the same way the
    # app does: bag-of-words in, intent probabilities out.
    model = keras.models.load_model(output_dir / "mymodel.h5")
    predictions = model.predict(train_x, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(train_y, axis=1)
    training_accuracy = float(np.mean(predicted_classes == true_classes))

    tests = [
        ("Give me study tips", "study_tips"),
        ("How can I study better", "study_tips"),
        ("Help me make a study plan", "study_tips"),
        ("How can I focus and study better for exams?", "study_tips"),
        ("Hello", "greeting"),
        ("Tell me a joke", "jokes"),
        ("Thank you", "thanks"),
        ("Who are you", "Identity"),
        ("What is the time", "datetime"),
        ("Bye", "goodbye"),
    ]

    test_results = []
    for sentence, expected in tests:
        probabilities = model.predict(
            create_bow(sentence, words, lemmatizer), verbose=0
        )[0]
        best = int(np.argmax(probabilities))
        test_results.append(
            {
                "sentence": sentence,
                "expected": expected,
                "predicted": classes[best],
                "probability": float(probabilities[best]),
                "above_original_0.8_threshold": bool(probabilities[best] > 0.8),
            }
        )

    summary = {
        "intents_in_file": len(intents["intents"]),
        "training_patterns": int(train_x.shape[0]),
        "vocabulary_words": len(words),
        "classes": classes,
        "class_count": len(classes),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "final_training_loss": float(history.history["loss"][-1]),
        "final_training_accuracy_from_fit": float(history.history["accuracy"][-1]),
        "reloaded_model_training_accuracy": training_accuracy,
        "tests": test_results,
    }
    (output_dir / "training-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
