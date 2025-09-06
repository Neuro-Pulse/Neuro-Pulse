# Configuration Guide

## Environment Variables

### Required Variables

```bash
# Solana Configuration
SOLANA_RPC_ENDPOINT=https://api.mainnet-beta.solana.com
SOLANA_WS_ENDPOINT=wss://api.mainnet-beta.solana.com

# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost:5432/neuropulse
REDIS_URL=redis://localhost:6379
MONGO_URI=mongodb://localhost:27017/neuropulse

# Security
JWT_SECRET=your-secret-key
API_KEY=your-api-key
```

### Optional Variables

```bash
# AI Configuration
AI_MODEL_PATH=/models/neural_predictor.onnx
CONFIDENCE_THRESHOLD=0.85

# Performance
MAX_CONCURRENT_RPC_CALLS=10
CACHE_TTL=3600
```

## Configuration Files

- `.env` - Environment variables
- `tsconfig.json` - TypeScript configuration
- `jest.config.js` - Test configuration
- `docker-compose.yml` - Docker services

## Feature Flags

Enable/disable features via environment:

```bash
ENABLE_AI_PREDICTIONS=true
ENABLE_REAL_TIME_TRACKING=true
ENABLE_HISTORICAL_ANALYSIS=true
```
