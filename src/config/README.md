# Config Module

Configuration management components.

## Components

- `ConfigManager` - Environment configuration management
- Configuration validation with Joi schemas
- Environment-specific settings

## Usage

```typescript
import { ConfigManager } from '@config/ConfigManager';

const config = new ConfigManager();
await config.load();

const value = config.get('SOLANA_RPC_ENDPOINT');
```
