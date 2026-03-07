# Backend Wiki

This page documents how to run the backend for this project with:
- Django + Uvicorn (ASGI)
- PostgreSQL in Docker
- Redis in Docker
- HTTP auth and WebSocket auth

## 1. Project Scope

This is backend-only setup for:
- Django project: `backend/`
- HTTP server: Uvicorn (`backend.asgi:application`)
- Database: Postgres from `docker-compose.yml`
- Channel layer: Redis from `docker-compose.yml`

## 2. Prerequisites

- Python 3.11+
- Docker + Docker Compose plugin (`docker compose`)
- A virtual environment

## 3. Install Backend Dependencies

From backend root:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements_backend.txt
```

Notes:
- `requirements_backend.txt` includes `channels`, `channels_redis`, `uvicorn`, `websockets`, `wsproto`.
- If you install only `requirements.txt`, WebSocket support is incomplete.

## 4. Start Infra (Postgres + Redis)

From backend root:

```bash
docker compose up -d db redis
```

Verify:

```bash
docker compose ps
docker compose logs -f redis
docker compose exec redis redis-cli ping
```

Expected for ping: `PONG`

## 5. Backend Runtime Configuration

Current project configuration (already present):
- DB host: `127.0.0.1:5432`
- Redis URL default: `redis://127.0.0.1:6379/0`
- ASGI app: `backend.asgi.application`
- WebSocket route: `/ws/notifications/`

Relevant files:
- `backend/settings.py`
- `backend/asgi.py`
- `backend/routing.py`
- `notification/websocket/routing.py`
- `auth2/ws_middleware.py`

## 6. Run Migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

## 7. Run Backend (ASGI)

Use Uvicorn, not `runserver`, for WebSocket verification:

```bash
uvicorn backend.asgi:application --host 127.0.0.1 --port 8000 --reload
```

## 8. HTTP Authentication (Get JWT)

Endpoint:
- `POST http://127.0.0.1:8000/auth/login`

Headers:
- `Content-Type: application/json`

Body:

```json
{
  "username": "your_username",
  "password": "your_password"
}
```

Sample:

```bash
curl -i -X POST http://127.0.0.1:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"your_username","password":"your_password"}'
```

## 9. WebSocket Authentication and Connect

WebSocket endpoint:
- `ws://127.0.0.1:8000/ws/notifications/`

Auth is enforced in `notification/websocket/consumer.py` via `scope["user"]`.

Supported auth methods in `auth2/ws_middleware.py`:
- Header: `Authorization: Bearer <JWT>`
- Query param: `?token=<JWT>`
- Query param: `?access_token=<JWT>`

Most reliable for Postman:

```text
ws://127.0.0.1:8000/ws/notifications/?token=<JWT>
```

Message payload must be valid JSON, for example:

```json
{"message":"hello"}
```

## 10. Postman Quick Flow

1. Create HTTP request to `/auth/login` and copy token.
2. Create WebSocket request to `/ws/notifications/?token=<JWT>`.
3. Connect.
4. Send JSON payload.

## 11. Troubleshooting

### 404 on `/ws/notifications/`

Cause:
- Request is treated as normal HTTP (WS handshake not active) or wrong server.

Fix:
- Start backend with Uvicorn ASGI command.
- Ensure URL is `ws://...`, not `http://...`.

### 403 on WebSocket handshake

Cause:
- User is anonymous (missing/invalid token).

Fix:
- Pass JWT in query param or `Authorization` header.
- Verify token from `/auth/login` is fresh.

### `No supported WebSocket library detected`

Cause:
- Uvicorn missing ws extras.

Fix:

```bash
pip install "uvicorn[standard]" websockets wsproto
```

### `JSONDecodeError` when sending WS message

Cause:
- Sent plain text instead of JSON.

Fix:
- Send valid JSON only.

### 400 from `/auth/login`

Common causes:
- Not using POST
- Invalid JSON body

## 12. Useful Commands

```bash
# backend checks
python manage.py check

# view docker logs
docker compose logs -f db
docker compose logs -f redis

# stop infra
docker compose down
```
