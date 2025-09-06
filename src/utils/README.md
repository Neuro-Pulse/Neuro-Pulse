# Utils Module

Utility functions and helper classes.

## Components

### Logger
Advanced logging system with Winston integration.

### MetricsCollector
Prometheus-based metrics collection.

### CacheManager
Multi-layer caching with Redis and memory support.

## Usage

```typescript
import { Logger } from '@utils/logger';
import { MetricsCollector } from '@utils/metrics';
import { CacheManager } from '@utils/cache';

const logger = new Logger('MyModule');
const metrics = new MetricsCollector('my_metrics');
const cache = new CacheManager('my_cache');
```
