# API Documentation

## REST API Endpoints

### Wallets

#### Track Wallet
```
POST /api/v1/wallets/track
{
  "address": "string",
  "label": "string"
}
```

#### Get Wallet Metrics
```
GET /api/v1/wallets/{address}/metrics
```

#### Analyze Wallet
```
POST /api/v1/wallets/{address}/analyze
```

### System

#### Health Check
```
GET /health
```

#### Metrics
```
GET /metrics
```

## WebSocket API

### Connection
```
wss://api.neuropulse.io/stream
```

### Events
- `wallet:activity`
- `signal:detected`
- `prediction:generated`

## GraphQL API

Endpoint: `/graphql`

Schema available at `/graphql/schema`
