import { readFileSync } from 'fs';
import * as dotenv from 'dotenv';
import * as joi from 'joi';

export class ConfigManager {
  private config: Record<string, any> = {};
  private schema: joi.ObjectSchema | null = null;

  constructor() {
    this.loadEnvironment();
    this.defineSchema();
  }

  private loadEnvironment(): void {
    dotenv.config();
    this.config = { ...process.env };
  }

  private defineSchema(): void {
    this.schema = joi.object({
      NODE_ENV: joi.string().valid('development', 'production', 'test').default('development'),
      PORT: joi.number().default(3000),
      SOLANA_RPC_ENDPOINT: joi.string().required(),
      SOLANA_WS_ENDPOINT: joi.string().optional(),
      DATABASE_URL: joi.string().required(),
      REDIS_URL: joi.string().required(),
      JWT_SECRET: joi.string().required(),
    }).unknown(true);
  }

  public async load(): Promise<void> {
    if (this.schema) {
      const { error, value } = this.schema.validate(this.config);
      if (error) {
        throw new Error(`Configuration validation failed: ${error.message}`);
      }
      this.config = value;
    }
  }

  public get(key: string): any {
    return this.config[key];
  }

  public set(key: string, value: any): void {
    this.config[key] = value;
  }

  public getAll(): Record<string, any> {
    return { ...this.config };
  }
}
