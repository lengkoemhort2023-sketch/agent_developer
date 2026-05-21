#!/bin/bash
set -euo pipefail

DATE=$(date +%Y%m%d_%H%M%S)
BASE="/mnt/back-up"
QDRANT_CONTAINER="agent_developer-qdrant-1"

echo "=== Backup started: $DATE ==="
echo ""

# Abort if less than 5 GB free on the backup volume
AVAIL=$(df -BG "$BASE" | awk 'NR==2 {gsub("G","",$4); print $4}')
if [ "$AVAIL" -lt 5 ]; then
    echo "ERROR: Only ${AVAIL}GB free on $BASE — need at least 5GB. Aborting."
    exit 1
fi

mkdir -p "$BASE/postgres" "$BASE/qdrant"

# Step 1: Django models + media
echo "[1/3] Backing up Django data (models + media)..."
docker exec agent_developer-web-1 python manage.py backup \
    --output-dir "$BASE" --compress
DJANGO_FILE=$(ls -t "$BASE"/backup_*.tar.gz 2>/dev/null | head -1)
if [ -z "$DJANGO_FILE" ] || [ ! -s "$DJANGO_FILE" ]; then
    echo "ERROR: Django backup file missing or empty."
    exit 1
fi
echo "  -> Saved to: $DJANGO_FILE"
echo ""

# Step 2: PostgreSQL raw dump
echo "[2/3] Backing up PostgreSQL..."
docker exec agent_developer-db-1 pg_dump -U postgres postgres \
    | gzip > "$BASE/postgres/db_$DATE.sql.gz"
if [ ! -s "$BASE/postgres/db_$DATE.sql.gz" ]; then
    echo "ERROR: PostgreSQL dump is empty — pg_dump may have failed."
    exit 1
fi
echo "  -> Saved to: $BASE/postgres/db_$DATE.sql.gz"
echo ""

# Step 3: Qdrant full-storage snapshot
# The API creates the snapshot inside the Qdrant container; we docker cp it out.
echo "[3/3] Backing up Qdrant..."
SNAPSHOT_RESPONSE=$(docker exec agent_developer-web-1 \
    curl -sf -X POST http://qdrant:6333/snapshots)
SNAPSHOT_NAME=$(echo "$SNAPSHOT_RESPONSE" \
    | sed 's/.*"name":"\([^"]*\)".*/\1/')
if [ -z "$SNAPSHOT_NAME" ] || [ "$SNAPSHOT_NAME" = "$SNAPSHOT_RESPONSE" ]; then
    echo "ERROR: Could not parse snapshot name. API response: $SNAPSHOT_RESPONSE"
    exit 1
fi
docker cp "$QDRANT_CONTAINER:/qdrant/snapshots/$SNAPSHOT_NAME" \
    "$BASE/qdrant/$SNAPSHOT_NAME"
if [ ! -s "$BASE/qdrant/$SNAPSHOT_NAME" ]; then
    echo "ERROR: Qdrant snapshot file missing or empty after copy."
    exit 1
fi
echo "  -> Saved to: $BASE/qdrant/$SNAPSHOT_NAME"
echo ""

echo "=== Backup complete ==="
echo "All backups saved to: $BASE"
ls -lh "$BASE/postgres/" "$BASE/qdrant/"
