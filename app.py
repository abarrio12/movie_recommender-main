"""
Aplicación Flask del recomendador de películas.

Carga el modelo generado por analysis.ipynb (model.pkl y similarity.pkl) y muestra
las películas más parecidas a la que elija el usuario, con sus pósters de TMDB.
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

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "390e76286265f7638bb6b19d86474639")
if not TMDB_API_KEY:
    log.warning("TMDB_API_KEY no está definida: se mostrarán las películas sin póster.")

N_RECOMMENDATIONS = 20
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"

# CAMBIO: rutas absolutas basadas en la carpeta de este archivo.
# POR QUÉ: con rutas relativas ('model.pkl') la app solo funciona si se arranca
# desde la carpeta del proyecto; así funciona se lance desde donde se lance.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------------------------

# CAMBIO: se usa `with open(...)` en lugar de pickle.load(open(...)).
# POR QUÉ: `with` cierra el archivo al terminar; la versión anterior lo dejaba abierto.
with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    movies = pickle.load(f)
with open(os.path.join(BASE_DIR, "similarity.pkl"), "rb") as f:
    similarity = pickle.load(f)

# CAMBIO: reset_index + comprobación de tamaños.
# POR QUÉ: la matriz de similitud se indexa por POSICIÓN (fila 0, 1, 2…). Si el índice
# del DataFrame tiene huecos (como pasaba tras el dropna del notebook original),
# la película buscada y su fila en la matriz no coinciden y las recomendaciones
# salen de otra película. El notebook ya lo corrige, pero lo repetimos aquí por
# seguridad, y comprobamos que ambos archivos corresponden al mismo modelo.
movies = movies.reset_index(drop=True)
if similarity.shape != (len(movies), len(movies)):
    raise RuntimeError(
        f"model.pkl ({len(movies)} películas) y similarity.pkl {similarity.shape} "
        "no coinciden. Vuelve a ejecutar analysis.ipynb para regenerar ambos."
    )

# CAMBIO: el desplegable usa el ID de TMDB como valor, y muestra "Título (año)".
# POR QUÉ: hay títulos repetidos que son películas distintas (Batman de 1966 y de 1989).
# Buscando por título siempre se elegía la primera; con el ID no hay ambigüedad.
id_to_position = pd.Series(movies.index, index=movies["id"])


def movie_label(row):
    return f"{row.title} ({row.year})" if pd.notna(row.year) else row.title


movie_options = [
    {"id": int(row.id), "label": movie_label(row)}
    for row in movies.sort_values("title").itertuples()
]

# ---------------------------------------------------------------------------
# Pósters
# ---------------------------------------------------------------------------

# CAMBIO: una única sesión HTTP reutilizada.
# POR QUÉ: reutiliza la conexión con TMDB en lugar de abrir una nueva en cada petición.
http = requests.Session()


# CAMBIO: caché de pósters.
# POR QUÉ: el póster de una película no cambia; si ya lo pedimos una vez, no hace falta
# volver a llamar a la API. Las películas populares salen en muchas recomendaciones.
# lru_cache no guarda las llamadas que lanzan una excepción, así que un fallo de red
# puntual no deja la película sin póster para siempre.
@lru_cache(maxsize=4096)
def _poster_url(movie_id):
    response = http.get(
        f"https://api.themoviedb.org/3/movie/{movie_id}",
        params={"api_key": TMDB_API_KEY, "language": "en-US"},
        # CAMBIO: timeout. POR QUÉ: sin él, si TMDB no responde la página se queda
        # colgada indefinidamente.
        timeout=5,
    )
    response.raise_for_status()
    # CAMBIO: .get() en lugar de ['poster_path'].
    # POR QUÉ: algunas películas no tienen póster (poster_path es None) y la versión
    # anterior fallaba al sumar un texto con None, rompiendo toda la página.
    poster_path = response.json().get("poster_path")
    return f"{POSTER_BASE_URL}{poster_path}" if poster_path else None


def fetch_poster(movie_id):
    """Devuelve la URL del póster, o None si no hay clave, no hay póster o falla la API.
    La plantilla muestra un recuadro con el título cuando recibe None."""
    if not TMDB_API_KEY:
        return None
    try:
        return _poster_url(int(movie_id))
    except requests.RequestException as error:
        # Se registra solo el tipo de error y el código HTTP, no el mensaje completo:
        # ese mensaje incluye la URL, y la URL incluye la API key.
        status = getattr(error.response, "status_code", None)
        log.warning("No se pudo obtener el póster de %s (%s, HTTP %s)",
                    movie_id, type(error).__name__, status)
        return None


# CAMBIO: los pósters se piden en paralelo (hasta 8 a la vez).
# POR QUÉ: antes se hacían 20 peticiones una detrás de otra; si cada una tarda
# 0,3 s, la página tardaba ~6 s. En paralelo tarda lo que la más lenta.
poster_pool = ThreadPoolExecutor(max_workers=8)

# ---------------------------------------------------------------------------
# Recomendaciones
# ---------------------------------------------------------------------------


def get_recommendations(position, n=N_RECOMMENDATIONS):
    """Devuelve las n películas más parecidas a la película en `position`."""
    scores = similarity[position]

    # CAMBIO: argpartition + argsort en lugar de ordenar las ~4800 películas.
    # POR QUÉ: solo necesitamos las n mejores. argpartition las separa sin ordenar
    # todo (más rápido) y luego ordenamos solo esas. Pedimos n + 1 porque la
    # película más parecida es ella misma, que después quitamos.
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
# Rutas
# ---------------------------------------------------------------------------


@app.route("/")
def home():
    return render_template("index.html", movie_options=movie_options)


# CAMBIO: la ruta acepta también GET.
# POR QUÉ: con solo POST, recargar la página de resultados muestra el aviso de
# "reenviar formulario". Con GET, la URL (/recommend?selected_movie=19995) se puede
# recargar, guardar o compartir.
@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    raw_id = request.values.get("selected_movie", "")

    # CAMBIO: validación de la entrada.
    # POR QUÉ: antes, un valor que no existiera provocaba un IndexError y un error 500.
    # Ahora se muestra un mensaje claro y la página sigue funcionando.
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
    # CAMBIO: debug desactivado por defecto.
    # POR QUÉ: el modo debug muestra el código y permite ejecutar Python desde el
    # navegador si hay un error; nunca debe estar activo en un servidor público.
    # Para desarrollar en local: FLASK_DEBUG=1 python app.py
    # En Docker la app no se arranca desde aquí, sino con gunicorn (ver Dockerfile).
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
