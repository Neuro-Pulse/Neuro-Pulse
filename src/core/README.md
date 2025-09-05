# Core Module

Core engine and system orchestration components.

## Components

### Engine
Main orchestration engine that manages all system components.

### DataPipeline
Multi-stage data processing pipeline with error handling.

### EventBus
Publish/subscribe messaging system for component communication.

### TaskScheduler
Cron-based task scheduling system.

## Architecture

The core module follows an event-driven architecture with loose coupling between components.

## Usage

```typescript
import { Engine } from '@core/Engine';

const engine = new Engine(config);
await engine.initialize();
await engine.start();
```
