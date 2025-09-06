import winston from 'winston';
import DailyRotateFile from 'winston-daily-rotate-file';
import { format } from 'winston';

export enum LogLevel {
  ERROR = 'error',
  WARN = 'warn',
  INFO = 'info',
  DEBUG = 'debug',
  VERBOSE = 'verbose',
}

export interface LogMetadata {
  timestamp: string;
  level: string;
  module: string;
  correlationId?: string;
  userId?: string;
  requestId?: string;
  [key: string]: any;
}

/**
 * Advanced logging system with multiple transports
 */
export class Logger {
  private readonly logger: winston.Logger;
  private readonly module: string;
  private static instances: Map<string, Logger> = new Map();

  constructor(module: string) {
    this.module = module;
    
    // Check if instance already exists
    if (Logger.instances.has(module)) {
      return Logger.instances.get(module)!;
    }
    
    this.logger = this.createLogger();
    Logger.instances.set(module, this);
  }

  /**
   * Create Winston logger instance
   */
  private createLogger(): winston.Logger {
    const logLevel = process.env.LOG_LEVEL || 'info';
    
    const customFormat = format.combine(
      format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss.SSS' }),
      format.errors({ stack: true }),
      format.metadata({ fillExcept: ['message', 'level', 'timestamp'] }),
      format.json(),
    );
    
    const consoleFormat = format.combine(
      format.colorize(),
      format.timestamp({ format: 'HH:mm:ss.SSS' }),
      format.printf(({ timestamp, level, message, metadata }) => {
        const meta = metadata && Object.keys(metadata).length 
          ? JSON.stringify(metadata) 
          : '';
        return `${timestamp} [${this.module}] ${level}: ${message} ${meta}`;
      }),
    );
    
    const transports: winston.transport[] = [
      // Console transport
      new winston.transports.Console({
        format: process.env.NODE_ENV === 'production' ? customFormat : consoleFormat,
        level: logLevel,
      }),
    ];
    
    // File transports for production
    if (process.env.NODE_ENV === 'production') {
      // Error logs
      transports.push(
        new DailyRotateFile({
          filename: 'logs/error-%DATE%.log',
          datePattern: 'YYYY-MM-DD',
          level: 'error',
          maxSize: '20m',
          maxFiles: '30d',
          format: customFormat,
        })
      );
      
      // Combined logs
      transports.push(
        new DailyRotateFile({
          filename: 'logs/combined-%DATE%.log',
          datePattern: 'YYYY-MM-DD',
          maxSize: '20m',
          maxFiles: '14d',
          format: customFormat,
        })
      );
    }
    
    return winston.createLogger({
      level: logLevel,
      format: customFormat,
      transports,
      exitOnError: false,
    });
  }

  /**
   * Log error message
   */
  public error(message: string, error?: Error | any, metadata?: any): void {
    const meta = this.enrichMetadata(metadata);
    
    if (error instanceof Error) {
      this.logger.error(message, {
        ...meta,
        error: {
          message: error.message,
          stack: error.stack,
          name: error.name,
        },
      });
    } else if (error) {
      this.logger.error(message, { ...meta, error });
    } else {
      this.logger.error(message, meta);
    }
  }

  /**
   * Log warning message
   */
  public warn(message: string, metadata?: any): void {
    this.logger.warn(message, this.enrichMetadata(metadata));
  }

  /**
   * Log info message
   */
  public info(message: string, metadata?: any): void {
    this.logger.info(message, this.enrichMetadata(metadata));
  }

  /**
   * Log debug message
   */
  public debug(message: string, metadata?: any): void {
    this.logger.debug(message, this.enrichMetadata(metadata));
  }

  /**
   * Log verbose message
   */
  public verbose(message: string, metadata?: any): void {
    this.logger.verbose(message, this.enrichMetadata(metadata));
  }

  /**
   * Create child logger with additional context
   */
  public child(metadata: any): Logger {
    const childLogger = new Logger(`${this.module}:child`);
    childLogger.logger.defaultMeta = { ...this.logger.defaultMeta, ...metadata };
    return childLogger;
  }

  /**
   * Enrich metadata with default values
   */
  private enrichMetadata(metadata?: any): any {
    return {
      module: this.module,
      pid: process.pid,
      ...metadata,
    };
  }

  /**
   * Get logger instance for module
   */
  public static getLogger(module: string): Logger {
    if (!Logger.instances.has(module)) {
      return new Logger(module);
    }
    return Logger.instances.get(module)!;
  }

  /**
   * Clear all logger instances
   */
  public static clearInstances(): void {
    Logger.instances.clear();
  }
}
