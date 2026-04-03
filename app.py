import json
import joblib
import numpy as np
from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

app = Flask(__name__)

# ── Load model and data ──────────────────────────────────────────────
print("Loading model...")
model = joblib.load("model/model_lr.pkl")

with open("model/model_config.json") as f:
    config = json.load(f)

with open("data/mood_articles.json", encoding="utf-8") as f:
    mood_articles = json.load(f)

print("Loading sentence transformer...")
embedder = SentenceTransformer(config["embedder"])
print("Ready!")

# ── Hybrid retrieval ─────────────────────────────────────────────────
def retrieve_articles(mood, query_text, top_n=5):
    candidates = mood_articles.get(mood, [])
    if not candidates:
        return []

    # BM25 keyword scoring
    corpus = [f"{a['title']} {a.get('description', '')}" for a in candidates]
    tokenized = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query_text.lower().split())

    # Semantic scoring
    query_vec = embedder.encode([query_text])[0]
    doc_vecs = embedder.encode(corpus)
    semantic_scores = np.dot(doc_vecs, query_vec) / (
        np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec) + 1e-8
    )

    # Normalize and combine
    bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
    sem_norm = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-8)
    hybrid = 0.4 * bm25_norm + 0.6 * sem_norm

    top_indices = np.argsort(hybrid)[::-1][:top_n]
    return [candidates[i] for i in top_indices]


# ── Routes ───────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    moods = sorted(mood_articles.keys())
    return render_template("index.html", moods=moods)


@app.route("/recommend", methods=["POST"])
def recommend():
    moods = sorted(mood_articles.keys())
    input_type = request.form.get("input_type")
    query_text = ""
    detected_mood = ""

    if input_type == "text":
        query_text = request.form.get("mood_text", "").strip()
        if not query_text:
            return render_template("index.html", moods=moods, error="Please enter some text.")
        query_vec = embedder.encode([query_text])
        detected_mood = model.predict(query_vec)[0]

    elif input_type == "dropdown":
        detected_mood = request.form.get("mood_dropdown", "")
        query_text = detected_mood
        if not detected_mood:
            return render_template("index.html", moods=moods, error="Please select a mood.")

    articles = retrieve_articles(detected_mood, query_text, top_n=5)

    return render_template(
        "index.html",
        moods=moods,
        detected_mood=detected_mood,
        query_text=query_text,
        articles=articles,
        input_type=input_type,
    )


#if __name__ == "__main__":
#    app.run(debug=True)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)