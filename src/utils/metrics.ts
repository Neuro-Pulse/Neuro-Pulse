import { EventEmitter } from 'events';
import * as client from 'prom-client';

export interface Metric {
  name: string;
  value: number;
  timestamp: number;
  labels?: Record<string, string>;
}

export interface MetricOptions {
  help: string;
  labelNames?: string[];
  buckets?: number[];
  percentiles?: number[];
}

/**
 * Comprehensive metrics collection system
 */
export class MetricsCollector extends EventEmitter {
  private readonly namespace: string;
  private readonly counters: Map<string, client.Counter>;
  private readonly gauges: Map<string, client.Gauge>;
  private readonly histograms: Map<string, client.Histogram>;
  private readonly summaries: Map<string, client.Summary>;
  private readonly register: client.Registry;

  constructor(namespace: string) {
    super();
    this.namespace = namespace;
    this.counters = new Map();
    this.gauges = new Map();
    this.histograms = new Map();
    this.summaries = new Map();
    this.register = new client.Registry();
    
    // Initialize default metrics
    this.initializeDefaultMetrics();
  }

  /**
   * Initialize default system metrics
   */
  private initializeDefaultMetrics(): void {
    // Collect default metrics
    client.collectDefaultMetrics({ 
      register: this.register,
      prefix: `${this.namespace}_`,
    });
    
    // Custom process metrics
    this.createGauge('process_memory_usage', {
      help: 'Process memory usage in bytes',
      labelNames: ['type'],
    });
    
    this.createCounter('events_total', {
      help: 'Total number of events processed',
      labelNames: ['event_type', 'status'],
    });
    
    this.createHistogram('response_time', {
      help: 'Response time in milliseconds',
      labelNames: ['endpoint', 'method'],
      buckets: [10, 50, 100, 200, 500, 1000, 2000, 5000],
    });
    
    // Start periodic collection
    this.startPeriodicCollection();
  }

  /**
   * Increment a counter metric
   */
  public increment(name: string, labels?: Record<string, string>, value: number = 1): void {
    try {
      const counter = this.getOrCreateCounter(name);
      
      if (labels) {
        counter.inc(labels, value);
      } else {
        counter.inc(value);
      }
      
      this.emit('metric:recorded', {
        type: 'counter',
        name,
        value,
        labels,
      });
    } catch (error) {
      this.emit('metric:error', { name, error });
    }
  }

  /**
   * Decrement a gauge metric
   */
  public decrement(name: string, labels?: Record<string, string>, value: number = 1): void {
    try {
      const gauge = this.getOrCreateGauge(name);
      
      if (labels) {
        gauge.dec(labels, value);
      } else {
        gauge.dec(value);
      }
      
      this.emit('metric:recorded', {
        type: 'gauge',
        name,
        value: -value,
        labels,
      });
    } catch (error) {
      this.emit('metric:error', { name, error });
    }
  }

  /**
   * Set a gauge metric
   */
  public set(name: string, value: number, labels?: Record<string, string>): void {
    try {
      const gauge = this.getOrCreateGauge(name);
      
      if (labels) {
        gauge.set(labels, value);
      } else {
        gauge.set(value);
      }
      
      this.emit('metric:recorded', {
        type: 'gauge',
        name,
        value,
        labels,
      });
    } catch (error) {
      this.emit('metric:error', { name, error });
    }
  }

  /**
   * Record a value for histogram/timing
   */
  public record(name: string, value: number | any, labels?: Record<string, string>): void {
    try {
      // Handle object values
      if (typeof value === 'object' && value !== null) {
        Object.entries(value).forEach(([key, val]) => {
          if (typeof val === 'number') {
            this.record(`${name}_${key}`, val, labels);
          }
        });
        return;
      }
      
      const histogram = this.getOrCreateHistogram(name);
      
      if (labels) {
        histogram.observe(labels, value);
      } else {
        histogram.observe(value);
      }
      
      this.emit('metric:recorded', {
        type: 'histogram',
        name,
        value,
        labels,
      });
    } catch (error) {
      this.emit('metric:error', { name, error });
    }
  }

  /**
   * Start a timer for measuring duration
   */
  public startTimer(name: string, labels?: Record<string, string>): () => void {
    const histogram = this.getOrCreateHistogram(name);
    const end = labels ? histogram.startTimer(labels) : histogram.startTimer();
    
    return () => {
      const duration = end();
      this.emit('metric:recorded', {
        type: 'histogram',
        name,
        value: duration,
        labels,
      });
      return duration;
    };
  }

  /**
   * Get all metrics in Prometheus format
   */
  public async getMetrics(): Promise<string> {
    return this.register.metrics();
  }

  /**
   * Get metrics as JSON
   */
  public async getMetricsAsJSON(): Promise<any> {
    return this.register.getMetricsAsJSON();
  }

  /**
   * Reset all metrics
   */
  public reset(): void {
    this.register.resetMetrics();
    this.emit('metrics:reset');
  }

  /**
   * Clear all metrics
   */
  public clear(): void {
    this.register.clear();
    this.counters.clear();
    this.gauges.clear();
    this.histograms.clear();
    this.summaries.clear();
    this.emit('metrics:cleared');
  }

  /**
   * Flush metrics (for external systems)
   */
  public async flush(): Promise<void> {
    const metrics = await this.getMetricsAsJSON();
    this.emit('metrics:flush', metrics);
    
    // Here you would send metrics to external systems
    // e.g., Prometheus Pushgateway, DataDog, etc.
  }

  /**
   * Create a counter metric
   */
  private createCounter(name: string, options: MetricOptions): client.Counter {
    const fullName = `${this.namespace}_${name}`;
    
    if (this.counters.has(fullName)) {
      return this.counters.get(fullName)!;
    }
    
    const counter = new client.Counter({
      name: fullName,
      help: options.help,
      labelNames: options.labelNames || [],
      registers: [this.register],
    });
    
    this.counters.set(fullName, counter);
    return counter;
  }

  /**
   * Create a gauge metric
   */
  private createGauge(name: string, options: MetricOptions): client.Gauge {
    const fullName = `${this.namespace}_${name}`;
    
    if (this.gauges.has(fullName)) {
      return this.gauges.get(fullName)!;
    }
    
    const gauge = new client.Gauge({
      name: fullName,
      help: options.help,
      labelNames: options.labelNames || [],
      registers: [this.register],
    });
    
    this.gauges.set(fullName, gauge);
    return gauge;
  }

  /**
   * Create a histogram metric
   */
  private createHistogram(name: string, options: MetricOptions): client.Histogram {
    const fullName = `${this.namespace}_${name}`;
    
    if (this.histograms.has(fullName)) {
      return this.histograms.get(fullName)!;
    }
    
    const histogram = new client.Histogram({
      name: fullName,
      help: options.help,
      labelNames: options.labelNames || [],
      buckets: options.buckets || [0.1, 0.5, 1, 2, 5, 10],
      registers: [this.register],
    });
    
    this.histograms.set(fullName, histogram);
    return histogram;
  }

  /**
   * Create a summary metric
   */
  private createSummary(name: string, options: MetricOptions): client.Summary {
    const fullName = `${this.namespace}_${name}`;
    
    if (this.summaries.has(fullName)) {
      return this.summaries.get(fullName)!;
    }
    
    const summary = new client.Summary({
      name: fullName,
      help: options.help,
      labelNames: options.labelNames || [],
      percentiles: options.percentiles || [0.5, 0.9, 0.95, 0.99],
      registers: [this.register],
    });
    
    this.summaries.set(fullName, summary);
    return summary;
  }

  /**
   * Get or create counter
   */
  private getOrCreateCounter(name: string): client.Counter {
    const fullName = `${this.namespace}_${name}`;
    
    if (!this.counters.has(fullName)) {
      return this.createCounter(name, {
        help: `Counter for ${name}`,
      });
    }
    
    return this.counters.get(fullName)!;
  }

  /**
   * Get or create gauge
   */
  private getOrCreateGauge(name: string): client.Gauge {
    const fullName = `${this.namespace}_${name}`;
    
    if (!this.gauges.has(fullName)) {
      return this.createGauge(name, {
        help: `Gauge for ${name}`,
      });
    }
    
    return this.gauges.get(fullName)!;
  }

  /**
   * Get or create histogram
   */
  private getOrCreateHistogram(name: string): client.Histogram {
    const fullName = `${this.namespace}_${name}`;
    
    if (!this.histograms.has(fullName)) {
      return this.createHistogram(name, {
        help: `Histogram for ${name}`,
      });
    }
    
    return this.histograms.get(fullName)!;
  }

  /**
   * Start periodic collection of system metrics
   */
  private startPeriodicCollection(): void {
    setInterval(() => {
      const memoryUsage = process.memoryUsage();
      
      this.set('process_memory_usage', memoryUsage.heapUsed, { type: 'heap_used' });
      this.set('process_memory_usage', memoryUsage.heapTotal, { type: 'heap_total' });
      this.set('process_memory_usage', memoryUsage.rss, { type: 'rss' });
      this.set('process_memory_usage', memoryUsage.external, { type: 'external' });
      
      const cpuUsage = process.cpuUsage();
      this.set('process_cpu_usage', cpuUsage.user, { type: 'user' });
      this.set('process_cpu_usage', cpuUsage.system, { type: 'system' });
      
      this.emit('metrics:collected');
    }, 10000); // Collect every 10 seconds
  }
}
