# Backend Wiki (Windows)

This runbook documents backend setup and runtime on Windows for:
- Django ASGI app (Uvicorn + Channels)
- Celery worker
- PostgreSQL + Redis via Docker Compose
- JWT auth for HTTP and WebSockets

## Table of Contents
- [1. Architecture](#1-architecture)
- [2. Prerequisites](#2-prerequisites)
- [3. Install Dependencies](#3-install-dependencies)
- [4. Start Infrastructure (Docker Desktop)](#4-start-infrastructure-docker-desktop)
- [5. Apply Migrations](#5-apply-migrations)
- [6. Run Services (3 Terminals)](#6-run-services-3-terminals)
- [7. Auth Quick Start (JWT)](#7-auth-quick-start-jwt)
- [8. WebSocket Endpoints](#8-websocket-endpoints)
- [9. Troubleshooting](#9-troubleshooting)
- [10. Useful Commands](#10-useful-commands)

## 1. Architecture

Backend runtime parts:
- **ASGI server**: `uvicorn backend.asgi:application`
- **HTTP + WebSocket app**: Django + Channels
- **Celery worker**: consumes long-running engine tasks
- **PostgreSQL**: Docker service `db`
- **Redis**: Docker service `redis` (broker/result backend + channel layer)

## 2. Prerequisites

- Windows 10/11
- Python 3.11+
- Docker Desktop (WSL2 backend recommended)
- Git

## 3. Install Dependencies

From backend root:

```powershell
python -m venv venv
```

Activate venv (PowerShell):

```powershell
.\venv\Scripts\Activate.ps1
```

If execution policy blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

Activate venv (CMD alternative):

```bat
venv\Scripts\activate.bat
```

Install dependencies:

```powershell
pip install -r requirements_backend.txt
```

## 4. Start Infrastructure (Docker Desktop)

Start services from backend root:

```powershell
docker compose up -d db redis
```

Verify:

```powershell
docker compose ps
docker compose logs -f db
docker compose logs -f redis
docker compose exec redis redis-cli ping
```

Expected ping: `PONG`

## 5. Apply Migrations

```powershell
python manage.py makemigrations
python manage.py migrate
```

## 6. Run Services (3 Terminals)

Use separate terminals for easier debugging.

### Terminal A: ASGI server (HTTP + WS)

```powershell
.\venv\Scripts\Activate.ps1
uvicorn backend.asgi:application --host 127.0.0.1 --port 8000 --reload
```

### Terminal B: Celery worker

```powershell
.\venv\Scripts\Activate.ps1
celery -A backend worker -l info --pool=solo
```

### Terminal C: Optional Docker logs

```powershell
docker compose logs -f db redis
```

## 7. Auth Quick Start (JWT)

Login endpoint:
- `POST http://127.0.0.1:8000/auth/login`

PowerShell sample:

```powershell
curl.exe -i -X POST http://127.0.0.1:8000/auth/login `
  -H "Content-Type: application/json" `
  -d "{\"username\":\"your_username\",\"password\":\"your_password\"}"
```

Use returned token:
- HTTP: `Authorization: Bearer <JWT>`
- WS query param: `?token=<JWT>`

## 8. WebSocket Endpoints

- Notifications:
  - `ws://127.0.0.1:8000/ws/notifications/?token=<JWT>`
- Engine status:
  - `ws://127.0.0.1:8000/ws/engine/status/<task_id>/?token=<JWT>`

## 9. Troubleshooting

### Port already in use (5432/6379/8000)

Fix:
- Stop conflicting process or change exposed ports in `docker-compose.yml`.

### Celery not receiving tasks (`PENDING` forever)

Checks:
- Redis container is running.
- Worker started from same backend folder with same env.
- Worker terminal shows task `received`.

### WS connects but no progress events

Checks:
- Celery worker is running.
- ASGI and worker use same Redis config.
- Watch both ASGI and worker logs for the same `task_id`.

### `403` during WS handshake

Cause:
- Invalid/missing JWT.

Fix:
- Pass fresh `?token=<JWT>`.

### Venv activation blocked

Use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 10. Useful Commands

```powershell
# Django checks
python manage.py check
python manage.py showmigrations

# Celery inspect
celery -A backend inspect ping
celery -A backend inspect active
celery -A backend inspect reserved

# Docker
docker compose ps
docker compose logs -f db
docker compose logs -f redis
docker compose down
```
