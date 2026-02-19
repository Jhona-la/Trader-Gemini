# 🐳 Docker Deployment Guide (OMEGA Level VIII)

## 1. Build the Image
The image is a multi-stage build that compiles `TA-Lib` and uses a slim Python runtime.

```bash
# Build with tag 'trader-gemini:latest'
docker build -t trader-gemini:latest .
```

## 2. Run the Container
You must pass your API keys as environment variables.

### Option A: Interactive Run (Best for testing)
```bash
docker run -it --rm \
  -e BINANCE_API_KEY="your_key" \
  -e BINANCE_SECRET_KEY="your_secret" \
  -e BOT_MODE="live" \
  trader-gemini:latest
```

### Option B: Detached Background Mode
```bash
docker run -d --name omega-bot \
  --restart unless-stopped \
  -e BINANCE_API_KEY="your_key" \
  -e BINANCE_SECRET_KEY="your_secret" \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  trader-gemini:latest
```

## 3. Persistent Data (Volumes)
The container saves logs and database files inside `/app/logs` and `/app/data`.
Mount these volumes (as shown in Option B) to persist data across container restarts.

## 4. Troubleshooting
If the container crashes immediately:
```bash
docker logs omega-bot
```
