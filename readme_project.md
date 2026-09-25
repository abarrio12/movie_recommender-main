# 🎬 Sistema de recomendación de películas

Recomienda películas parecidas a la que elijas, usando filtrado basado en contenido
(sinopsis, géneros, palabras clave, reparto y director) sobre el dataset TMDB 5000.
Se sirve con Flask y se empaqueta con Docker.

Como bonus, el notebook incluye un **filtrado colaborativo basado en usuarios** y un
**recomendador híbrido**, evaluados con el dataset MovieLens.

## Estructura

```
movie-recommender/
├── analysis.ipynb        # limpieza, modelo, evaluación y bonus; genera los .pkl
├── app.py                # aplicación Flask
├── model.pkl             # películas (id, título, año)      ← lo genera el notebook
├── similarity.pkl        # matriz de similitud (float32)    ← lo genera el notebook
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── templates/
│   └── index.html
└── data/
    ├── tmdb_5000_movies.csv
    ├── tmdb_5000_credits.csv
    └── ml-latest-small/  # MovieLens; el notebook lo descarga solo
```

## Cómo ejecutarlo

### 1. Generar el modelo

```bash
pip install -r requirements.txt
jupyter notebook analysis.ipynb     # ejecutar todas las celdas
```

Esto crea `model.pkl` y `similarity.pkl`.

### 2. Clave de TMDB (para los pósters)

Crea una clave gratuita en <https://www.themoviedb.org/settings/api>.
Sin clave la app funciona, pero muestra las películas sin póster.

### 3a. En local

```bash
export TMDB_API_KEY="tu_clave"      # Windows PowerShell: $env:TMDB_API_KEY="tu_clave"
python app.py
```

### 3b. Con Docker

```bash
docker build -t movie-recommender .
docker run -p 5000:5000 -e TMDB_API_KEY="tu_clave" movie-recommender
```

Después abre <http://localhost:5000>.

## Resultados

**Modelo de contenido**: TF-IDF con 5000 términos. Se eligió frente a CountVectorizer porque,
evaluado con valoraciones reales de MovieLens, recomienda con más del doble de precisión.

**Recomendación para usuarios de MovieLens** (precision@10 / recall@10 en test):

| Modelo                         | precision@10 | recall@10 |
|--------------------------------|-------------:|----------:|
| Popularidad (referencia)       | 0.114        | 0.116     |
| Contenido puro                 | 0.024        | 0.037     |
| Colaborativo (user-based, k=60)| 0.127        | 0.135     |
| **Híbrido (α = 0.75)**         | **0.143**    | **0.157** |

El colaborativo también mejora la predicción de notas: RMSE 0.89 frente a 0.95 de predecir
la media de cada usuario.

## Notas

- `similarity.pkl` ocupa ~92 MB: GitHub lo acepta (límite de 100 MB por archivo) pero avisa
  a partir de 50 MB. Si da problemas al subirlo, usa [Git LFS](https://git-lfs.com/).
- Los `.pkl` deben generarse con las mismas versiones de pandas/numpy que usa la app
  (las de `requirements.txt`); si no, pueden no cargarse dentro de Docker.
