import { Engine } from './core/Engine';
import { Logger } from './utils/logger';

const logger = new Logger('Main');

async function main(): Promise<void> {
  try {
    logger.info('Starting NeuroPulse system');
    
    const engine = new Engine({
      mode: process.env.NODE_ENV as 'production' | 'development' | 'test' || 'development',
      workers: 4,
      maxConcurrency: 100,
      heartbeatInterval: 10000,
      shutdownTimeout: 30000,
    });
    
    await engine.initialize();
    await engine.start();
    
    logger.info('NeuroPulse system started successfully');
  } catch (error) {
    logger.error('Failed to start NeuroPulse', error);
    process.exit(1);
  }
}

// Handle uncaught errors
process.on('uncaughtException', (error) => {
  logger.error('Uncaught exception', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason) => {
  logger.error('Unhandled rejection', reason);
  process.exit(1);
});

// Start the application
main().catch((error) => {
  logger.error('Fatal error', error);
  process.exit(1);
});
