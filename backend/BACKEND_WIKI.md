# Backend Wiki

This runbook documents backend setup and runtime for:
- Django ASGI app (Uvicorn + Channels)
- Celery worker
- PostgreSQL + Redis via Docker Compose
- JWT auth for HTTP and WebSockets

## Table of Contents
- [1. Architecture](#1-architecture)
- [2. Prerequisites](#2-prerequisites)
- [3. Install Dependencies](#3-install-dependencies)
- [4. Start Infrastructure (Docker)](#4-start-infrastructure-docker)
- [5. Apply Migrations](#5-apply-migrations)
- [6. Run Services (3 Terminals)](#6-run-services-3-terminals)
- [7. Auth Quick Start (JWT)](#7-auth-quick-start-jwt)
- [8. WebSocket Endpoints](#8-websocket-endpoints)
- [9. End-to-End Engine Flow](#9-end-to-end-engine-flow)
- [10. Troubleshooting](#10-troubleshooting)
- [11. Useful Commands](#11-useful-commands)

## 1. Architecture

Backend runtime parts:
- **ASGI server**: `uvicorn backend.asgi:application`
- **HTTP + WebSocket app**: Django + Channels
- **Celery worker**: consumes long-running engine tasks
- **PostgreSQL**: primary DB (`db` service)
- **Redis**: broker/result backend + channel layer (`redis` service)

Relevant files:
- `backend/settings.py`
- `backend/asgi.py`
- `backend/celery.py`
- `backend/routing.py`
- `auth2/ws_middleware.py`
- `core/tasks.py`
- `docker-compose.yml`

## 2. Prerequisites

- Python 3.11+
- Docker + Docker Compose plugin (`docker compose`)
- Virtual environment tooling

## 3. Install Dependencies

From backend root:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements_backend.txt
```

Notes:
- Use `requirements_backend.txt` for Channels/Celery/WebSocket runtime.
- Installing only `requirements.txt` may miss runtime dependencies.

## 4. Start Infrastructure (Docker)

The current `docker-compose.yml` defines:
- `db` (Postgres)
- `redis` (Redis)

Start:

```bash
docker compose up -d db redis
```

Verify:

```bash
docker compose ps
docker compose logs -f db
docker compose logs -f redis
docker compose exec redis redis-cli ping
```

Expected ping response: `PONG`

## 5. Apply Migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

## 6. Run Services (3 Terminals)

Run each part separately so logs are easy to debug.

### Terminal A: ASGI server (HTTP + WS)

```bash
source venv/bin/activate
uvicorn backend.asgi:application --host 127.0.0.1 --port 8000 --reload
```

### Terminal B: Celery worker

```bash
source venv/bin/activate
celery -A backend worker -l info --pool=solo
```

### Terminal C: Optional infra logs

```bash
docker compose logs -f db redis
```

## 7. Auth Quick Start (JWT)

Login endpoint:
- `POST http://127.0.0.1:8000/auth/login`

Sample:

```bash
curl -i -X POST http://127.0.0.1:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"your_username","password":"your_password"}'
```

Use the returned token for HTTP (`Authorization: Bearer <JWT>`) and WS (`?token=<JWT>`).

## 8. WebSocket Endpoints

### Notifications
- `ws://127.0.0.1:8000/ws/notifications/?token=<JWT>`

### Engine task realtime status
- `ws://127.0.0.1:8000/ws/engine/status/<task_id>/?token=<JWT>`

WS auth is handled in `auth2/ws_middleware.py`.

Supported auth input:
- `Authorization: Bearer <JWT>` header
- `?token=<JWT>` query param

## 9. End-to-End Engine Flow

1. Call `POST /core/engine/trigger`.
2. Receive `task_id` and `ws_path`.
3. Subscribe to `/ws/engine/status/<task_id>/`.
4. Celery processes task and emits progress/success/failure events.
5. Frontend may fallback to `GET /core/engine/status/<task_id>`.

## 10. Troubleshooting

### WS connects but no progress events

Checks:
- Celery worker is running in a separate terminal.
- Worker uses same `.env` / Redis settings as ASGI.
- Triggered task appears in worker logs (`received`, `started`, `succeeded/failed`).

### `PENDING` forever on engine status

Cause:
- Worker not consuming tasks or wrong broker URL.

Fix:
- Restart worker.
- Verify Redis is up and reachable.

### 404 on WS routes

Cause:
- Wrong protocol or wrong server mode.

Fix:
- Use `ws://`, not `http://`.
- Use Uvicorn ASGI command (not `runserver` for WS verification).

### 403 on WS handshake

Cause:
- Missing/invalid JWT.

Fix:
- Pass fresh token via `?token=<JWT>`.

### Missing WebSocket runtime libs

```bash
pip install "uvicorn[standard]" websockets wsproto
```

## 11. Useful Commands

```bash
# Django checks
python manage.py check

# Migrations status
python manage.py showmigrations

# Inspect celery worker
celery -A backend inspect ping
celery -A backend inspect active
celery -A backend inspect reserved

# Infra logs
docker compose logs -f db
docker compose logs -f redis

# Stop infra
docker compose down
```
