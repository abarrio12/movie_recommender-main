FROM python:3.12-slim

# CAMBIO: se ha quitado la instalación de build-essential, gcc y python3-dev.
# POR QUÉ: todas las librerías de requirements.txt (incluida nltk) se instalan ya
# compiladas (wheels), así que no hace falta compilador. Quitarlo hace la imagen
# varios cientos de MB más pequeña y el build bastante más rápido.

# No generar archivos .pyc y mostrar los logs al momento
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# CAMBIO: primero se copia solo requirements.txt y se instalan las dependencias;
# el código se copia después.
# POR QUÉ: Docker guarda cada paso en caché. Antes se copiaba todo el proyecto
# primero, así que cualquier cambio en app.py obligaba a reinstalar todas las
# librerías. Ahora, si requirements.txt no cambia, ese paso se reutiliza.
# (requirements.txt incluye nltk.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# CAMBIO: se copian solo los archivos que necesita la app (antes `COPY . /app`).
# POR QUÉ: el notebook y los CSV no hacen falta para servir la web y harían la
# imagen más grande. El .dockerignore refuerza esto.
COPY app.py model.pkl similarity.pkl ./
COPY templates/ templates/

# Usuario sin permisos de administrador (buena práctica de seguridad)
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

EXPOSE 5000

# CAMBIO: gunicorn en lugar de `flask run`.
# POR QUÉ: `flask run` es un servidor de desarrollo y la propia documentación de
# Flask dice que no se use en producción. gunicorn atiende varias peticiones a la vez.
#   --workers 2 --threads 4 : 2 procesos con 4 hilos cada uno
#   --preload               : carga similarity.pkl (~92 MB) una sola vez y los
#                             procesos la comparten, en lugar de cargarla dos veces
#   --timeout 60            : margen para pedir los pósters a TMDB
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "--preload", "--timeout", "60", "app:app"]
