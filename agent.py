import json
import ollama
from pathlib import Path
from rapidfuzz import fuzz

BOOKS_PATH = Path(__file__).parent / "books.json"
MODEL = "llama3.2"


def load_books() -> dict:
    with open(BOOKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def match_topic(ocr_text: str) -> dict:
    """
    OCR metnini books.json ile karşılaştırır,
    en uygun konuyu ve system prompt'u döner.
    """
    data = load_books()
    best_score = 0
    best_book = None

    for book in data["books"]:
        for keyword in book["keywords"]:
            score = fuzz.partial_ratio(keyword, ocr_text)
            if score > best_score:
                best_score = score
                best_book = book

    # Eşik: %50 altıysa bilinmeyen kitap
    if best_score < 50 or best_book is None:
        return {
            "topic": "genel",
            "system_prompt": data["default_system_prompt"],
            "score": best_score,
        }

    return {
        "topic": best_book["topic"],
        "system_prompt": best_book["system_prompt"],
        "score": best_score,
    }


class BookAgent:
    def __init__(self, system_prompt: str, topic: str):
        self.topic = topic
        self.system_prompt = system_prompt
        self.history: list[dict] = []

    def chat(self, user_message: str) -> str:
        """Kullanıcı mesajını ajanla işler, yanıt döner."""
        self.history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": self.system_prompt}] + self.history

        response = ollama.chat(model=MODEL, messages=messages)
        assistant_reply = response["message"]["content"]

        self.history.append({"role": "assistant", "content": assistant_reply})
        return assistant_reply

    def reset(self):
        """Yeni kitap için geçmişi temizle."""
        self.history = []
