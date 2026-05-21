#!/bin/sh
set -e

APP_ROOT="${APP_ROOT:-/usr/src/app}"
export PYTHONPATH="$APP_ROOT:$PYTHONPATH"

QDRANT_HOST="${QDRANT_HOST:-qdrant}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
DJANGO_RUNSERVER_HOST="${DJANGO_RUNSERVER_HOST:-127.0.0.1}"
DJANGO_RUNSERVER_PORT="${DJANGO_RUNSERVER_PORT:-8000}"

# Wait for PostgreSQL
if [ "$DATABASE" = "postgres" ]; then
    echo "[INFO] Waiting for PostgreSQL..."
    while ! nc -z $DB_HOST $DB_PORT; do
        sleep 0.1
    done
    echo "[INFO] PostgreSQL started"
fi

# Wait for Qdrant
echo "[INFO] Waiting for Qdrant..."
while ! nc -z "$QDRANT_HOST" "$QDRANT_PORT"; do
    sleep 0.5
done
echo "[INFO] Qdrant started"

# Check if we should run celery or web server
if echo "$@" | grep -q "worker"; then
    echo "[INFO] Starting Celery worker..."
    exec env PYTHONPATH="$APP_ROOT:$PYTHONPATH" "$@"
elif echo "$@" | grep -q "beat"; then
    echo "[INFO] Running migrations for Celery beat..."
    python manage.py migrate --noinput
    echo "[INFO] Starting Celery beat..."
    exec env PYTHONPATH="$APP_ROOT:$PYTHONPATH" "$@"
else
    # Collect static files
    echo "[INFO] Collecting static files..."
    python manage.py collectstatic --noinput

    # Run migrations
    echo "[INFO] Running migrations..."
    python manage.py migrate

    if [ "${RUN_GUNICORN:-true}" = "true" ] || [ "${RUN_GUNICORN:-true}" = "1" ]; then
        GUNICORN_BIND="${GUNICORN_BIND:-127.0.0.1:${DJANGO_RUNSERVER_PORT}}"
        GUNICORN_WORKERS="${GUNICORN_WORKERS:-1}"
        GUNICORN_THREADS="${GUNICORN_THREADS:-8}"
        GUNICORN_TIMEOUT="${GUNICORN_TIMEOUT:-180}"

        echo "[INFO] Starting Gunicorn: bind=${GUNICORN_BIND}, workers=${GUNICORN_WORKERS}, threads=${GUNICORN_THREADS}"
        exec gunicorn \
            app.core.wsgi:application \
            --bind "${GUNICORN_BIND}" \
            --worker-class gthread \
            --workers "${GUNICORN_WORKERS}" \
            --threads "${GUNICORN_THREADS}" \
            --timeout "${GUNICORN_TIMEOUT}"
    fi

    echo "[INFO] Starting Django development server..."
    exec python manage.py runserver "${DJANGO_RUNSERVER_HOST}:${DJANGO_RUNSERVER_PORT}"
fi
