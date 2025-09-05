import { EventEmitter } from 'events';
import { Logger } from '@utils/logger';

export class EventBus extends EventEmitter {
  private logger: Logger;
  private channels: Map<string, Set<Function>> = new Map();

  constructor() {
    super();
    this.logger = new Logger('EventBus');
  }

  public async initialize(): Promise<void> {
    this.logger.info('Event bus initialized');
  }

  public async publish(event: any): Promise<void> {
    const channel = event.channel || 'default';
    this.emit(channel, event);
    
    const subscribers = this.channels.get(channel);
    if (subscribers) {
      for (const callback of subscribers) {
        try {
          await callback(event);
        } catch (error) {
          this.logger.error(`Subscriber error on channel ${channel}`, error);
        }
      }
    }
  }

  public subscribe(channel: string, callback: Function): void {
    if (!this.channels.has(channel)) {
      this.channels.set(channel, new Set());
    }
    this.channels.get(channel)!.add(callback);
  }

  public unsubscribe(channel: string, callback: Function): void {
    const subscribers = this.channels.get(channel);
    if (subscribers) {
      subscribers.delete(callback);
    }
  }

  public async shutdown(): Promise<void> {
    this.channels.clear();
    this.removeAllListeners();
    this.logger.info('Event bus shutdown');
  }
}
