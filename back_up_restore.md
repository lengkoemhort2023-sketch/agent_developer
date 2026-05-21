# Backup & Restore Guide

## Overview

Three data stores are backed up to `/mnt/back-up/`:

| Data | Method | Format |
|---|---|---|
| Django models + media | `python manage.py backup` | `.tar.gz` (JSON + files) |
| PostgreSQL | `pg_dump` | `.sql.gz` |
| Qdrant vectors | Snapshot API | `.snapshot` |

---

## Backup

### Source project: `/home/sysadmin/agent_developer/`

Run manually anytime:

```bash
cd /home/sysadmin/agent_developer
./backup_all.sh
```

This creates three outputs in `/mnt/back-up/`:

```
/mnt/back-up/
├── backup_<YYYYMMDD_HHMMSS>.tar.gz        ← Django models + media
├── postgres/
│   └── db_<YYYYMMDD_HHMMSS>.sql.gz        ← PostgreSQL dump
└── qdrant/
    ├── full-snapshot-<timestamp>.snapshot  ← Qdrant vectors
    └── <collection>/
        └── *.snapshot                      ← Per-collection snapshots
```

### What gets backed up

**Django models** (as JSON):
- `user.User`
- `chat.ChatSession`, `chat.ChatMessage`, `chat.ChatInput`
- `document.Document`
- `department.Department`
- `docs_type.DocumentType`

**Media files**: everything under `MEDIA_ROOT`

---

## Restore

### Target project: `/home/sa-koemhort/code/agent_developer/`

```bash
cd /home/sa-koemhort/code/agent_developer
./restore_all.sh <YYYYMMDD_HHMMSS>
```

Replace `<YYYYMMDD_HHMMSS>` with the timestamp of the backup to restore.

**Example:**

```bash
./restore_all.sh 20260517_203000
```

### What the restore does

1. Starts `db` + `qdrant` containers
2. Restores PostgreSQL from `db_<timestamp>.sql.gz`
3. Starts `web` container
4. Restores Django models + media from `backup_<timestamp>.tar.gz`
5. Restores Qdrant vectors from the full snapshot
6. Starts remaining services (`celery`, `beat`, `redis`)

### Restore manually (step by step)

**PostgreSQL:**

```bash
gunzip -c /mnt/back-up/postgres/db_<timestamp>.sql.gz \
  | docker exec -i agent_developer_alt-db-1 psql -U postgres postgres
```

**Django data:**

```bash
docker cp /mnt/back-up/backup_<timestamp>.tar.gz agent_developer_alt-web-1:/tmp/
docker exec agent_developer_alt-web-1 python manage.py restore /tmp/backup_<timestamp>.tar.gz
```

**Qdrant:**

```bash
docker exec agent_developer_alt-web-1 curl -s -X PUT \
  "http://qdrant:6333/snapshots/full-snapshot-<timestamp>.snapshot"
```

---

## Files

### Source project (`/home/sysadmin/agent_developer/`)

| File | Purpose |
|---|---|
| `backup_all.sh` | Run this to create a backup |
| `docker-compose.yml` | Has bind mounts to `/mnt/back-up/` |

### Target project (`/home/sa-koemhort/code/agent_developer/`)

| File | Purpose |
|---|---|
| `restore_all.sh` | Run this to restore from a backup |
| `docker-compose.yml` | Has bind mounts to `/mnt/back-up/` |

### Django management commands

Both source and target projects have these in `management_commands/management/commands/`:

| Command | File | Description |
|---|---|---|
| `python manage.py backup` | `management/commands/backup.py` | Dumps models to JSON + copies media |
| `python manage.py restore` | `management/commands/restore.py` | Restores models from JSON + media files |
