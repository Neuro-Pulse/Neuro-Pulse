import { EventEmitter } from 'events';
import Redis from 'ioredis';
import { Logger } from './logger';

export interface CacheOptions {
  ttl?: number;
  namespace?: string;
  maxKeys?: number;
  checkPeriod?: number;
}

export interface CacheEntry {
  key: string;
  value: any;
  ttl?: number;
  createdAt: number;
  accessedAt: number;
  hits: number;
}

/**
 * Advanced caching system with Redis and in-memory support
 */
export class CacheManager extends EventEmitter {
  private readonly namespace: string;
  private readonly logger: Logger;
  private readonly memoryCache: Map<string, CacheEntry>;
  private readonly redis: Redis | null;
  private readonly defaultTTL: number;
  private readonly maxKeys: number;
  private readonly checkPeriod: number;
  private cleanupTimer: NodeJS.Timeout | null = null;

  constructor(namespace: string, options: CacheOptions = {}) {
    super();
    this.namespace = options.namespace || namespace;
    this.logger = new Logger(`CacheManager:${this.namespace}`);
    this.memoryCache = new Map();
    this.defaultTTL = options.ttl || 3600; // 1 hour default
    this.maxKeys = options.maxKeys || 10000;
    this.checkPeriod = options.checkPeriod || 600; // 10 minutes
    
    // Initialize Redis if available
    this.redis = this.initializeRedis();
    
    // Start cleanup process
    this.startCleanup();
  }

  /**
   * Initialize Redis connection
   */
  private initializeRedis(): Redis | null {
    try {
      const redisUrl = process.env.REDIS_URL;
      if (!redisUrl) {
        this.logger.warn('Redis URL not configured, using memory cache only');
        return null;
      }
      
      const redis = new Redis(redisUrl, {
        retryStrategy: (times) => Math.min(times * 50, 2000),
        maxRetriesPerRequest: 3,
        enableReadyCheck: true,
        lazyConnect: true,
      });
      
      redis.on('connect', () => {
        this.logger.info('Redis connected');
        this.emit('redis:connected');
      });
      
      redis.on('error', (error) => {
        this.logger.error('Redis error', error);
        this.emit('redis:error', error);
      });
      
      redis.on('close', () => {
        this.logger.warn('Redis connection closed');
        this.emit('redis:closed');
      });
      
      // Connect to Redis
      redis.connect().catch((error) => {
        this.logger.error('Failed to connect to Redis', error);
      });
      
      return redis;
    } catch (error) {
      this.logger.error('Failed to initialize Redis', error);
      return null;
    }
  }

  /**
   * Get value from cache
   */
  public async get<T = any>(key: string): Promise<T | null> {
    const fullKey = this.getFullKey(key);
    
    try {
      // Try memory cache first
      const memoryEntry = this.memoryCache.get(fullKey);
      if (memoryEntry) {
        if (this.isExpired(memoryEntry)) {
          this.memoryCache.delete(fullKey);
        } else {
          memoryEntry.accessedAt = Date.now();
          memoryEntry.hits++;
          this.emit('cache:hit', { key, source: 'memory' });
          return memoryEntry.value as T;
        }
      }
      
      // Try Redis if available
      if (this.redis) {
        const value = await this.redis.get(fullKey);
        if (value) {
          try {
            const parsed = JSON.parse(value);
            
            // Store in memory cache for faster access
            this.setMemoryCache(fullKey, parsed, this.defaultTTL);
            
            this.emit('cache:hit', { key, source: 'redis' });
            return parsed as T;
          } catch (error) {
            this.logger.error('Failed to parse Redis value', { key, error });
          }
        }
      }
      
      this.emit('cache:miss', { key });
      return null;
    } catch (error) {
      this.logger.error('Cache get error', { key, error });
      this.emit('cache:error', { key, error });
      return null;
    }
  }

  /**
   * Set value in cache
   */
  public async set(key: string, value: any, ttl?: number): Promise<boolean> {
    const fullKey = this.getFullKey(key);
    const ttlSeconds = ttl || this.defaultTTL;
    
    try {
      // Store in memory cache
      this.setMemoryCache(fullKey, value, ttlSeconds);
      
      // Store in Redis if available
      if (this.redis) {
        const serialized = JSON.stringify(value);
        await this.redis.setex(fullKey, ttlSeconds, serialized);
      }
      
      this.emit('cache:set', { key, ttl: ttlSeconds });
      return true;
    } catch (error) {
      this.logger.error('Cache set error', { key, error });
      this.emit('cache:error', { key, error });
      return false;
    }
  }

  /**
   * Delete value from cache
   */
  public async delete(key: string): Promise<boolean> {
    const fullKey = this.getFullKey(key);
    
    try {
      // Delete from memory cache
      this.memoryCache.delete(fullKey);
      
      // Delete from Redis if available
      if (this.redis) {
        await this.redis.del(fullKey);
      }
      
      this.emit('cache:delete', { key });
      return true;
    } catch (error) {
      this.logger.error('Cache delete error', { key, error });
      this.emit('cache:error', { key, error });
      return false;
    }
  }

  /**
   * Check if key exists
   */
  public async has(key: string): Promise<boolean> {
    const fullKey = this.getFullKey(key);
    
    try {
      // Check memory cache
      if (this.memoryCache.has(fullKey)) {
        const entry = this.memoryCache.get(fullKey)!;
        if (!this.isExpired(entry)) {
          return true;
        }
        this.memoryCache.delete(fullKey);
      }
      
      // Check Redis if available
      if (this.redis) {
        const exists = await this.redis.exists(fullKey);
        return exists === 1;
      }
      
      return false;
    } catch (error) {
      this.logger.error('Cache has error', { key, error });
      return false;
    }
  }

  /**
   * Clear all cache entries
   */
  public async clear(): Promise<void> {
    try {
      // Clear memory cache
      this.memoryCache.clear();
      
      // Clear Redis entries if available
      if (this.redis) {
        const pattern = `${this.namespace}:*`;
        const keys = await this.redis.keys(pattern);
        
        if (keys.length > 0) {
          await this.redis.del(...keys);
        }
      }
      
      this.emit('cache:cleared');
      this.logger.info('Cache cleared');
    } catch (error) {
      this.logger.error('Cache clear error', error);
      this.emit('cache:error', { error });
    }
  }

  /**
   * Get cache statistics
   */
  public getStats(): {
    memoryEntries: number;
    memorySize: number;
    hits: number;
    misses: number;
    hitRate: number;
  } {
    let totalHits = 0;
    let memorySize = 0;
    
    this.memoryCache.forEach((entry) => {
      totalHits += entry.hits;
      memorySize += JSON.stringify(entry.value).length;
    });
    
    return {
      memoryEntries: this.memoryCache.size,
      memorySize,
      hits: totalHits,
      misses: 0, // Would need to track this separately
      hitRate: 0, // Would need to calculate based on hits/misses
    };
  }

  /**
   * Get or set cache value
   */
  public async getOrSet<T = any>(
    key: string,
    factory: () => Promise<T> | T,
    ttl?: number,
  ): Promise<T> {
    // Try to get from cache
    const cached = await this.get<T>(key);
    if (cached !== null) {
      return cached;
    }
    
    // Generate value
    const value = await factory();
    
    // Store in cache
    await this.set(key, value, ttl);
    
    return value;
  }

  /**
   * Memoize a function
   */
  public memoize<T extends (...args: any[]) => any>(
    fn: T,
    keyGenerator?: (...args: Parameters<T>) => string,
    ttl?: number,
  ): T {
    const generateKey = keyGenerator || ((...args) => JSON.stringify(args));
    
    return (async (...args: Parameters<T>) => {
      const key = `memoize:${fn.name}:${generateKey(...args)}`;
      
      return this.getOrSet(
        key,
        () => fn(...args),
        ttl,
      );
    }) as T;
  }

  /**
   * Set value in memory cache
   */
  private setMemoryCache(key: string, value: any, ttl: number): void {
    // Check max keys limit
    if (this.memoryCache.size >= this.maxKeys) {
      this.evictOldest();
    }
    
    const entry: CacheEntry = {
      key,
      value,
      ttl,
      createdAt: Date.now(),
      accessedAt: Date.now(),
      hits: 0,
    };
    
    this.memoryCache.set(key, entry);
  }

  /**
   * Check if cache entry is expired
   */
  private isExpired(entry: CacheEntry): boolean {
    if (!entry.ttl) return false;
    
    const expiresAt = entry.createdAt + (entry.ttl * 1000);
    return Date.now() > expiresAt;
  }

  /**
   * Evict oldest entries
   */
  private evictOldest(): void {
    const entries = Array.from(this.memoryCache.entries());
    
    // Sort by last accessed time
    entries.sort((a, b) => a[1].accessedAt - b[1].accessedAt);
    
    // Remove 10% of oldest entries
    const toRemove = Math.ceil(this.maxKeys * 0.1);
    
    for (let i = 0; i < toRemove && i < entries.length; i++) {
      this.memoryCache.delete(entries[i][0]);
    }
    
    this.emit('cache:eviction', { removed: toRemove });
  }

  /**
   * Get full cache key with namespace
   */
  private getFullKey(key: string): string {
    return `${this.namespace}:${key}`;
  }

  /**
   * Start cleanup process
   */
  private startCleanup(): void {
    this.cleanupTimer = setInterval(() => {
      this.cleanup();
    }, this.checkPeriod * 1000);
  }

  /**
   * Cleanup expired entries
   */
  private cleanup(): void {
    let removed = 0;
    
    for (const [key, entry] of this.memoryCache.entries()) {
      if (this.isExpired(entry)) {
        this.memoryCache.delete(key);
        removed++;
      }
    }
    
    if (removed > 0) {
      this.logger.debug(`Cleaned up ${removed} expired cache entries`);
      this.emit('cache:cleanup', { removed });
    }
  }

  /**
   * Dispose cache manager
   */
  public async dispose(): Promise<void> {
    // Stop cleanup timer
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer);
      this.cleanupTimer = null;
    }
    
    // Clear memory cache
    this.memoryCache.clear();
    
    // Disconnect Redis
    if (this.redis) {
      await this.redis.quit();
    }
    
    this.removeAllListeners();
    this.logger.info('Cache manager disposed');
  }
}
