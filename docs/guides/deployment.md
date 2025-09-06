# Deployment Guide

## Docker Deployment

### Build Image
```bash
docker build -t neuropulse:latest .
```

### Run Container
```bash
docker run -d \
  --name neuropulse \
  -p 3000:3000 \
  -p 8080:8080 \
  --env-file .env \
  neuropulse:latest
```

## Docker Compose

```bash
docker-compose up -d
```

## Kubernetes Deployment

### Apply Manifests
```bash
kubectl apply -f k8s/
```

### Scale Deployment
```bash
kubectl scale deployment neuropulse --replicas=3
```

## Environment Variables

Required environment variables:
- `SOLANA_RPC_ENDPOINT`
- `DATABASE_URL`
- `REDIS_URL`
- `JWT_SECRET`

## Health Checks

Monitor application health:
```bash
curl http://localhost:3000/health
```
