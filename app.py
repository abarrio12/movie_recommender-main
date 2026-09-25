"""
Flask app for the movie recommender.

Loads the model produced by analysis.ipynb (model.pkl and similarity.pkl) and shows
the movies most similar to the one the user picks, with their TMDB posters.
"""
import logging
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
from flask import Flask, render_template, request

import truststore
truststore.inject_into_ssl()   # use Windows' certificate store (fixes SSL errors behind antivirus/proxies)


app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# The TMDB API key is read from the TMDB_API_KEY environment variable:
#     export TMDB_API_KEY="your_key"                      (locally)
#     docker run -e TMDB_API_KEY="your_key" ...           (in Docker)
# If it isn't set, the key provided with the course template is used as a fallback.
# WHY an environment variable: keys should not live in code that is pushed to GitHub.
# The fallback is only acceptable here because that key is already public in the
# course repository; never hard-code a personal or paid key like this.
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "390e76286265f7638bb6b19d86474639")
if not TMDB_API_KEY:
    log.warning("TMDB_API_KEY is not set: movies will be shown without posters.")

N_RECOMMENDATIONS = 20
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"

# CHANGE: absolute paths based on this file's folder.
# WHY: with relative paths ('model.pkl') the app only works when started from the
# project folder; this way it works wherever it is launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

# CHANGE: `with open(...)` instead of pickle.load(open(...)).
# WHY: `with` closes the file when done; the previous version left it open.
with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    movies = pickle.load(f)
with open(os.path.join(BASE_DIR, "similarity.pkl"), "rb") as f:
    similarity = pickle.load(f)

# CHANGE: reset_index + size check.
# WHY: the similarity matrix is indexed by POSITION (row 0, 1, 2...). If the DataFrame
# index has gaps (as it did after the dropna in the original notebook), the selected
# movie and its row in the matrix don't match, and the recommendations come from a
# different movie. The notebook already fixes this, but we repeat it here to be safe,
# and we check that both files belong to the same model.
movies = movies.reset_index(drop=True)
if similarity.shape != (len(movies), len(movies)):
    raise RuntimeError(
        f"model.pkl ({len(movies)} movies) and similarity.pkl {similarity.shape} "
        "don't match. Re-run analysis.ipynb to regenerate both."
    )

# CHANGE: the dropdown uses the TMDB ID as its value and shows "Title (year)".
# WHY: some titles are shared by different movies (Batman from 1966 and 1989).
# Looking up by title always picked the first one; with the ID there's no ambiguity.
id_to_position = pd.Series(movies.index, index=movies["id"])


def movie_label(row):
    return f"{row.title} ({row.year})" if pd.notna(row.year) else row.title


movie_options = [
    {"id": int(row.id), "label": movie_label(row)}
    for row in movies.sort_values("title").itertuples()
]

# ---------------------------------------------------------------------------
# Posters
# ---------------------------------------------------------------------------

# CHANGE: a single reused HTTP session.
# WHY: it reuses the connection to TMDB instead of opening a new one for every request.
http = requests.Session()


# CHANGE: poster cache.
# WHY: a movie's poster doesn't change; once we've fetched it there's no need to call
# the API again. Popular movies show up in many recommendation lists.
# lru_cache does not store calls that raise an exception, so a one-off network
# failure doesn't leave the movie without a poster forever.
@lru_cache(maxsize=4096)
def _poster_url(movie_id):
    response = http.get(
        f"https://api.themoviedb.org/3/movie/{movie_id}",
        params={"api_key": TMDB_API_KEY, "language": "en-US"},
        # CHANGE: timeout. WHY: without it, if TMDB doesn't respond the page hangs forever.
        timeout=5,
    )
    response.raise_for_status()
    # CHANGE: .get() instead of ['poster_path'].
    # WHY: some movies have no poster (poster_path is None), and the previous version
    # crashed when adding a string to None, breaking the whole page.
    poster_path = response.json().get("poster_path")
    return f"{POSTER_BASE_URL}{poster_path}" if poster_path else None


def fetch_poster(movie_id):
    """Return the poster URL, or None if there's no key, no poster or the API fails.
    The template shows a box with the title when it receives None."""
    if not TMDB_API_KEY:
        return None
    try:
        return _poster_url(int(movie_id))
    except requests.RequestException as error:
        # Only the error type and HTTP status are logged, not the full message:
        # that message contains the URL, and the URL contains the API key.
        status = getattr(error.response, "status_code", None)
        log.warning("Could not fetch the poster for %s (%s, HTTP %s)",
                    movie_id, type(error).__name__, status)
        return None


# CHANGE: posters are fetched in parallel (up to 8 at a time).
# WHY: previously 20 requests were made one after another; at ~0.3 s each, the page
# took ~6 s. In parallel it takes as long as the slowest one.
poster_pool = ThreadPoolExecutor(max_workers=8)

# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


def get_recommendations(position, n=N_RECOMMENDATIONS):
    """Return the n movies most similar to the movie at `position`."""
    scores = similarity[position]

    # CHANGE: argpartition + argsort instead of sorting all ~4800 movies.
    # WHY: we only need the top n. argpartition separates them without sorting
    # everything (faster), and then we sort only those. We take n + 1 because the
    # most similar movie is the movie itself, which we then remove.
    candidates = np.argpartition(-scores, n + 1)[: n + 1]
    candidates = candidates[np.argsort(-scores[candidates])]
    top = [i for i in candidates if i != position][:n]

    recommended = movies.iloc[top]
    posters = list(poster_pool.map(fetch_poster, recommended["id"]))
    return [
        {"title": row.title, "year": row.year if pd.notna(row.year) else None, "poster": poster}
        for row, poster in zip(recommended.itertuples(), posters)
    ]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def home():
    return render_template("index.html", movie_options=movie_options)


# CHANGE: the route also accepts GET.
# WHY: with POST only, reloading the results page shows the "resubmit form" warning.
# With GET, the URL (/recommend?selected_movie=19995) can be reloaded, bookmarked or shared.
@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    raw_id = request.values.get("selected_movie", "")

    # CHANGE: input validation.
    # WHY: previously, a value that didn't exist caused an IndexError and a 500 error.
    # Now a clear message is shown and the page keeps working.
    try:
        selected_id = int(raw_id)
        position = int(id_to_position[selected_id])
    except (ValueError, KeyError):
        return render_template(
            "index.html",
            movie_options=movie_options,
            error="That movie isn't in the list. Pick one from the dropdown.",
        ), 400

    return render_template(
        "index.html",
        movie_options=movie_options,
        selected_id=selected_id,
        selected_label=movie_label(movies.iloc[position]),
        recommendations=get_recommendations(position),
    )


if __name__ == "__main__":
    # CHANGE: debug is off by default.
    # WHY: debug mode shows the source code and lets anyone run Python from the browser
    # when an error occurs; it must never be enabled on a public server.
    # For local development: FLASK_DEBUG=1 python app.py
    # In Docker the app isn't started from here but with gunicorn (see Dockerfile).
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
