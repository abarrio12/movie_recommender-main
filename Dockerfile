FROM python:3.12-slim

# CHANGE: removed the installation of build-essential, gcc and python3-dev.
# WHY: every library in requirements.txt (including nltk) installs as a pre-built
# wheel, so no compiler is needed. Removing it makes the image several hundred MB
# smaller and the build much faster.

# Don't write .pyc files, and show logs immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# CHANGE: copy only requirements.txt first and install the dependencies;
# the code is copied afterwards.
# WHY: Docker caches each step. Previously the whole project was copied first, so
# any change to app.py forced a reinstall of every library. Now, if requirements.txt
# hasn't changed, this step is reused from the cache.
# (requirements.txt includes nltk.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# CHANGE: copy only the files the app needs (previously `COPY . /app`).
# WHY: the notebook and the CSV files aren't needed to serve the website and would
# make the image bigger. The .dockerignore reinforces this.
COPY app.py model.pkl similarity.pkl ./
COPY templates/ templates/

# Non-root user (security best practice)
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

EXPOSE 5000

# CHANGE: gunicorn instead of `flask run`.
# WHY: `flask run` is a development server, and Flask's own documentation says not
# to use it in production. gunicorn handles several requests at the same time.
#   --workers 2 --threads 4 : 2 processes with 4 threads each
#   --preload               : loads similarity.pkl (~92 MB) once and the processes
#                             share it, instead of loading it twice
#   --timeout 60            : leaves time to fetch the posters from TMDB
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "--preload", "--timeout", "60", "app:app"]
