import { EventEmitter } from 'events';
import * as cron from 'node-cron';
import { Logger } from '@utils/logger';

export interface ScheduledTask {
  name: string;
  schedule: string;
  handler: () => Promise<void>;
  task?: cron.ScheduledTask;
}

export class TaskScheduler extends EventEmitter {
  private tasks: Map<string, ScheduledTask> = new Map();
  private logger: Logger;
  private isRunning: boolean = false;

  constructor() {
    super();
    this.logger = new Logger('TaskScheduler');
  }

  public async initialize(): Promise<void> {
    this.logger.info('Task scheduler initialized');
  }

  public addTask(name: string, schedule: string, handler: () => Promise<void>): void {
    const task = cron.schedule(schedule, async () => {
      try {
        await handler();
        this.emit('task:complete', { name });
      } catch (error) {
        this.logger.error(`Task ${name} failed`, error);
        this.emit('task:error', { name, error });
      }
    }, { scheduled: false });

    this.tasks.set(name, { name, schedule, handler, task });
    this.logger.info(`Added scheduled task: ${name} (${schedule})`);
  }

  public async start(): Promise<void> {
    for (const [name, task] of this.tasks) {
      if (task.task) {
        task.task.start();
        this.logger.info(`Started task: ${name}`);
      }
    }
    this.isRunning = true;
  }

  public async stop(): Promise<void> {
    for (const [name, task] of this.tasks) {
      if (task.task) {
        task.task.stop();
        this.logger.info(`Stopped task: ${name}`);
      }
    }
    this.isRunning = false;
  }

  public async pause(): Promise<void> {
    await this.stop();
  }

  public async resume(): Promise<void> {
    await this.start();
  }
}
