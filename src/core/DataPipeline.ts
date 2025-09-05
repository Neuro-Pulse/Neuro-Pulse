import { EventEmitter } from 'events';
import { Logger } from '@utils/logger';

export type PipelineStage = (data: any) => Promise<any>;

export class DataPipeline extends EventEmitter {
  private stages: Map<string, PipelineStage> = new Map();
  private logger: Logger;
  private isRunning: boolean = false;

  constructor() {
    super();
    this.logger = new Logger('DataPipeline');
  }

  public async initialize(): Promise<void> {
    this.logger.info('Data pipeline initialized');
  }

  public addStage(name: string, handler: PipelineStage): void {
    this.stages.set(name, handler);
    this.logger.info(`Added pipeline stage: ${name}`);
  }

  public async process(data: any): Promise<any> {
    let result = data;
    
    for (const [name, handler] of this.stages) {
      const startTime = Date.now();
      
      try {
        result = await handler(result);
        
        this.emit('stage:complete', {
          name,
          duration: Date.now() - startTime,
        });
      } catch (error) {
        this.logger.error(`Pipeline stage ${name} failed`, error);
        this.emit('error', { stage: name, error });
        throw error;
      }
    }
    
    return result;
  }

  public async start(): Promise<void> {
    this.isRunning = true;
    this.logger.info('Data pipeline started');
  }

  public async stop(): Promise<void> {
    this.isRunning = false;
    this.logger.info('Data pipeline stopped');
  }

  public async pause(): Promise<void> {
    this.isRunning = false;
    this.logger.info('Data pipeline paused');
  }

  public async resume(): Promise<void> {
    this.isRunning = true;
    this.logger.info('Data pipeline resumed');
  }
}
