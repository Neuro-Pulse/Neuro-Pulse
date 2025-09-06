import { EventEmitter } from 'events';
import * as ss from 'simple-statistics';
import { Logger } from '@utils/logger';
import { MetricsCollector } from '@utils/metrics';

export interface SignalData {
  timestamp: number;
  value: number;
  volume?: number;
  metadata?: Record<string, unknown>;
}

export interface ProcessedSignal {
  original: SignalData[];
  processed: number[];
  features: SignalFeatures;
  patterns: PatternMatch[];
  anomalies: AnomalyPoint[];
}

export interface SignalFeatures {
  mean: number;
  variance: number;
  standardDeviation: number;
  skewness: number;
  kurtosis: number;
  autocorrelation: number[];
  spectralEntropy: number;
  dominantFrequency: number;
  powerSpectralDensity: number[];
  waveletCoefficients: number[][];
}

export interface PatternMatch {
  type: PatternType;
  confidence: number;
  startIndex: number;
  endIndex: number;
  parameters: Record<string, number>;
}

export interface AnomalyPoint {
  index: number;
  value: number;
  zscore: number;
  probability: number;
  type: 'spike' | 'dip' | 'level_shift' | 'variance_change';
}

export enum PatternType {
  TREND_UP = 'trend_up',
  TREND_DOWN = 'trend_down',
  CONSOLIDATION = 'consolidation',
  BREAKOUT = 'breakout',
  REVERSAL = 'reversal',
  DOUBLE_TOP = 'double_top',
  DOUBLE_BOTTOM = 'double_bottom',
  HEAD_SHOULDERS = 'head_shoulders',
  TRIANGLE = 'triangle',
  FLAG = 'flag',
}

/**
 * Advanced signal processing for wallet activity patterns
 */
export class SignalProcessor extends EventEmitter {
  private readonly logger: Logger;
  private readonly metrics: MetricsCollector;
  private readonly windowSize: number;
  private readonly overlapRatio: number;
  private readonly fftPoints: number;

  constructor(
    windowSize: number = 60,
    overlapRatio: number = 0.5,
    fftPoints: number = 512,
  ) {
    super();
    this.logger = new Logger('SignalProcessor');
    this.metrics = new MetricsCollector('ai.signal_processor');
    this.windowSize = windowSize;
    this.overlapRatio = overlapRatio;
    this.fftPoints = fftPoints;
  }

  /**
   * Process raw signal data
   */
  public async processSignal(data: SignalData[]): Promise<ProcessedSignal> {
    const startTime = Date.now();
    
    try {
      this.logger.debug(`Processing signal with ${data.length} data points`);
      
      // Extract values
      const values = data.map(d => d.value);
      
      // Apply preprocessing
      const processed = await this.preprocess(values);
      
      // Extract features
      const features = await this.extractFeatures(processed);
      
      // Detect patterns
      const patterns = await this.detectPatterns(processed);
      
      // Detect anomalies
      const anomalies = await this.detectAnomalies(processed);
      
      const result: ProcessedSignal = {
        original: data,
        processed,
        features,
        patterns,
        anomalies,
      };
      
      const processingTime = Date.now() - startTime;
      this.metrics.record('processing_time', processingTime);
      this.emit('signal:processed', result);
      
      return result;
    } catch (error) {
      this.logger.error('Signal processing failed', error);
      this.metrics.increment('processing_errors');
      throw error;
    }
  }

  /**
   * Preprocess signal data
   */
  private async preprocess(values: number[]): Promise<number[]> {
    // Remove outliers
    const cleaned = this.removeOutliers(values);
    
    // Apply smoothing
    const smoothed = this.applySmoothingFilter(cleaned);
    
    // Normalize
    const normalized = this.normalize(smoothed);
    
    // Detrend
    const detrended = this.detrend(normalized);
    
    return detrended;
  }

  /**
   * Extract comprehensive signal features
   */
  private async extractFeatures(signal: number[]): Promise<SignalFeatures> {
    const features: SignalFeatures = {
      mean: ss.mean(signal),
      variance: ss.variance(signal),
      standardDeviation: ss.standardDeviation(signal),
      skewness: this.calculateSkewness(signal),
      kurtosis: this.calculateKurtosis(signal),
      autocorrelation: this.calculateAutocorrelation(signal),
      spectralEntropy: this.calculateSpectralEntropy(signal),
      dominantFrequency: this.findDominantFrequency(signal),
      powerSpectralDensity: this.calculatePSD(signal),
      waveletCoefficients: this.waveletTransform(signal),
    };
    
    this.metrics.record('features', features);
    return features;
  }

  /**
   * Detect patterns in the signal
   */
  private async detectPatterns(signal: number[]): Promise<PatternMatch[]> {
    const patterns: PatternMatch[] = [];
    
    // Trend detection
    const trends = this.detectTrends(signal);
    patterns.push(...trends);
    
    // Chart pattern detection
    const chartPatterns = this.detectChartPatterns(signal);
    patterns.push(...chartPatterns);
    
    // Cycle detection
    const cycles = this.detectCycles(signal);
    patterns.push(...cycles);
    
    this.logger.info(`Detected ${patterns.length} patterns`);
    this.metrics.record('patterns_detected', patterns.length);
    
    return patterns;
  }

  /**
   * Detect anomalies in the signal
   */
  private async detectAnomalies(signal: number[]): Promise<AnomalyPoint[]> {
    const anomalies: AnomalyPoint[] = [];
    const mean = ss.mean(signal);
    const std = ss.standardDeviation(signal);
    
    // Z-score based detection
    for (let i = 0; i < signal.length; i++) {
      const zscore = Math.abs((signal[i] - mean) / std);
      
      if (zscore > 3) {
        const anomaly: AnomalyPoint = {
          index: i,
          value: signal[i],
          zscore,
          probability: this.calculateAnomalyProbability(zscore),
          type: this.classifyAnomaly(signal, i),
        };
        anomalies.push(anomaly);
      }
    }
    
    // Advanced anomaly detection using isolation forest
    const isolationAnomalies = this.isolationForest(signal);
    anomalies.push(...isolationAnomalies);
    
    this.logger.info(`Detected ${anomalies.length} anomalies`);
    this.metrics.record('anomalies_detected', anomalies.length);
    
    return anomalies;
  }

  /**
   * Remove outliers using IQR method
   */
  private removeOutliers(values: number[]): number[] {
    const q1 = ss.quantile(values, 0.25);
    const q3 = ss.quantile(values, 0.75);
    const iqr = q3 - q1;
    const lowerBound = q1 - 1.5 * iqr;
    const upperBound = q3 + 1.5 * iqr;
    
    return values.map(v => {
      if (v < lowerBound) return lowerBound;
      if (v > upperBound) return upperBound;
      return v;
    });
  }

  /**
   * Apply smoothing filter
   */
  private applySmoothingFilter(values: number[], windowSize: number = 5): number[] {
    const result: number[] = [];
    const halfWindow = Math.floor(windowSize / 2);
    
    for (let i = 0; i < values.length; i++) {
      const start = Math.max(0, i - halfWindow);
      const end = Math.min(values.length, i + halfWindow + 1);
      const window = values.slice(start, end);
      result.push(ss.mean(window));
    }
    
    return result;
  }

  /**
   * Normalize signal to [0, 1] range
   */
  private normalize(values: number[]): number[] {
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min;
    
    if (range === 0) return values.map(() => 0.5);
    
    return values.map(v => (v - min) / range);
  }

  /**
   * Remove linear trend from signal
   */
  private detrend(values: number[]): number[] {
    const x = Array.from({ length: values.length }, (_, i) => i);
    const regression = ss.linearRegression(x.map((xi, i) => [xi, values[i]]));
    const trend = x.map(xi => regression.m * xi + regression.b);
    
    return values.map((v, i) => v - trend[i]);
  }

  /**
   * Calculate skewness of the signal
   */
  private calculateSkewness(values: number[]): number {
    const mean = ss.mean(values);
    const std = ss.standardDeviation(values);
    const n = values.length;
    
    const m3 = values.reduce((sum, v) => sum + Math.pow(v - mean, 3), 0) / n;
    return m3 / Math.pow(std, 3);
  }

  /**
   * Calculate kurtosis of the signal
   */
  private calculateKurtosis(values: number[]): number {
    const mean = ss.mean(values);
    const std = ss.standardDeviation(values);
    const n = values.length;
    
    const m4 = values.reduce((sum, v) => sum + Math.pow(v - mean, 4), 0) / n;
    return m4 / Math.pow(std, 4) - 3;
  }

  /**
   * Calculate autocorrelation function
   */
  private calculateAutocorrelation(values: number[], maxLag: number = 20): number[] {
    const acf: number[] = [];
    const mean = ss.mean(values);
    const variance = ss.variance(values);
    
    for (let lag = 0; lag <= maxLag; lag++) {
      let sum = 0;
      for (let i = 0; i < values.length - lag; i++) {
        sum += (values[i] - mean) * (values[i + lag] - mean);
      }
      acf.push(sum / ((values.length - lag) * variance));
    }
    
    return acf;
  }

  /**
   * Calculate spectral entropy
   */
  private calculateSpectralEntropy(values: number[]): number {
    const psd = this.calculatePSD(values);
    const totalPower = psd.reduce((sum, p) => sum + p, 0);
    
    if (totalPower === 0) return 0;
    
    const normalizedPSD = psd.map(p => p / totalPower);
    const entropy = -normalizedPSD
      .filter(p => p > 0)
      .reduce((sum, p) => sum + p * Math.log2(p), 0);
    
    return entropy / Math.log2(psd.length);
  }

  /**
   * Find dominant frequency using FFT
   */
  private findDominantFrequency(values: number[]): number {
    const fft = this.performFFT(values);
    const magnitudes = fft.map(c => Math.sqrt(c.real * c.real + c.imag * c.imag));
    
    let maxMagnitude = 0;
    let dominantIndex = 0;
    
    for (let i = 1; i < magnitudes.length / 2; i++) {
      if (magnitudes[i] > maxMagnitude) {
        maxMagnitude = magnitudes[i];
        dominantIndex = i;
      }
    }
    
    return dominantIndex / values.length;
  }

  /**
   * Calculate Power Spectral Density
   */
  private calculatePSD(values: number[]): number[] {
    const fft = this.performFFT(values);
    return fft.map(c => (c.real * c.real + c.imag * c.imag) / values.length);
  }

  /**
   * Perform Fast Fourier Transform
   */
  private performFFT(values: number[]): Array<{ real: number; imag: number }> {
    const n = values.length;
    const result: Array<{ real: number; imag: number }> = [];
    
    for (let k = 0; k < n; k++) {
      let real = 0;
      let imag = 0;
      
      for (let t = 0; t < n; t++) {
        const angle = -2 * Math.PI * k * t / n;
        real += values[t] * Math.cos(angle);
        imag += values[t] * Math.sin(angle);
      }
      
      result.push({ real, imag });
    }
    
    return result;
  }

  /**
   * Wavelet transform for multi-resolution analysis
   */
  private waveletTransform(values: number[]): number[][] {
    const levels = 4;
    const coefficients: number[][] = [];
    let current = [...values];
    
    for (let level = 0; level < levels; level++) {
      const { approximation, detail } = this.dwt(current);
      coefficients.push(detail);
      current = approximation;
      
      if (current.length < 2) break;
    }
    
    return coefficients;
  }

  /**
   * Discrete Wavelet Transform (Haar wavelet)
   */
  private dwt(values: number[]): { approximation: number[]; detail: number[] } {
    const approximation: number[] = [];
    const detail: number[] = [];
    
    for (let i = 0; i < values.length - 1; i += 2) {
      approximation.push((values[i] + values[i + 1]) / Math.sqrt(2));
      detail.push((values[i] - values[i + 1]) / Math.sqrt(2));
    }
    
    return { approximation, detail };
  }

  /**
   * Detect trends in the signal
   */
  private detectTrends(values: number[]): PatternMatch[] {
    const patterns: PatternMatch[] = [];
    const windowSize = Math.min(20, Math.floor(values.length / 5));
    
    for (let i = 0; i <= values.length - windowSize; i++) {
      const window = values.slice(i, i + windowSize);
      const x = Array.from({ length: windowSize }, (_, j) => j);
      const regression = ss.linearRegression(x.map((xi, j) => [xi, window[j]]));
      
      const slope = regression.m;
      const r2 = ss.rSquared(
        x.map((xi, j) => [xi, window[j]]),
        x => regression.m * x + regression.b
      );
      
      if (Math.abs(slope) > 0.01 && r2 > 0.7) {
        patterns.push({
          type: slope > 0 ? PatternType.TREND_UP : PatternType.TREND_DOWN,
          confidence: r2,
          startIndex: i,
          endIndex: i + windowSize - 1,
          parameters: { slope, intercept: regression.b, r2 },
        });
      }
    }
    
    return this.mergeOverlappingPatterns(patterns);
  }

  /**
   * Detect chart patterns
   */
  private detectChartPatterns(values: number[]): PatternMatch[] {
    const patterns: PatternMatch[] = [];
    
    // Double top/bottom detection
    patterns.push(...this.detectDoubleTopsBottoms(values));
    
    // Head and shoulders detection
    patterns.push(...this.detectHeadShoulders(values));
    
    // Triangle pattern detection
    patterns.push(...this.detectTriangles(values));
    
    return patterns;
  }

  /**
   * Detect double tops and bottoms
   */
  private detectDoubleTopsBottoms(values: number[]): PatternMatch[] {
    const patterns: PatternMatch[] = [];
    const peaks = this.findPeaks(values);
    const troughs = this.findTroughs(values);
    
    // Double tops
    for (let i = 0; i < peaks.length - 1; i++) {
      const peak1 = peaks[i];
      const peak2 = peaks[i + 1];
      const similarity = 1 - Math.abs(values[peak1] - values[peak2]) / Math.max(values[peak1], values[peak2]);
      
      if (similarity > 0.95 && peak2 - peak1 > 10) {
        patterns.push({
          type: PatternType.DOUBLE_TOP,
          confidence: similarity,
          startIndex: peak1,
          endIndex: peak2,
          parameters: { peak1Value: values[peak1], peak2Value: values[peak2] },
        });
      }
    }
    
    // Double bottoms
    for (let i = 0; i < troughs.length - 1; i++) {
      const trough1 = troughs[i];
      const trough2 = troughs[i + 1];
      const similarity = 1 - Math.abs(values[trough1] - values[trough2]) / Math.max(values[trough1], values[trough2]);
      
      if (similarity > 0.95 && trough2 - trough1 > 10) {
        patterns.push({
          type: PatternType.DOUBLE_BOTTOM,
          confidence: similarity,
          startIndex: trough1,
          endIndex: trough2,
          parameters: { trough1Value: values[trough1], trough2Value: values[trough2] },
        });
      }
    }
    
    return patterns;
  }

  /**
   * Detect head and shoulders pattern
   */
  private detectHeadShoulders(values: number[]): PatternMatch[] {
    const patterns: PatternMatch[] = [];
    const peaks = this.findPeaks(values);
    
    for (let i = 0; i < peaks.length - 2; i++) {
      const leftShoulder = peaks[i];
      const head = peaks[i + 1];
      const rightShoulder = peaks[i + 2];
      
      const leftHeight = values[leftShoulder];
      const headHeight = values[head];
      const rightHeight = values[rightShoulder];
      
      if (headHeight > leftHeight && headHeight > rightHeight) {
        const shoulderSimilarity = 1 - Math.abs(leftHeight - rightHeight) / Math.max(leftHeight, rightHeight);
        
        if (shoulderSimilarity > 0.9) {
          patterns.push({
            type: PatternType.HEAD_SHOULDERS,
            confidence: shoulderSimilarity,
            startIndex: leftShoulder,
            endIndex: rightShoulder,
            parameters: {
              leftShoulderHeight: leftHeight,
              headHeight,
              rightShoulderHeight: rightHeight,
            },
          });
        }
      }
    }
    
    return patterns;
  }

  /**
   * Detect triangle patterns
   */
  private detectTriangles(values: number[]): PatternMatch[] {
    const patterns: PatternMatch[] = [];
    const windowSize = Math.min(30, Math.floor(values.length / 3));
    
    for (let i = 0; i <= values.length - windowSize; i++) {
      const window = values.slice(i, i + windowSize);
      const peaks = this.findPeaks(window);
      const troughs = this.findTroughs(window);
      
      if (peaks.length >= 2 && troughs.length >= 2) {
        const peakTrend = this.calculateTrend(peaks.map(p => window[p]));
        const troughTrend = this.calculateTrend(troughs.map(t => window[t]));
        
        if (Math.abs(peakTrend) < 0.1 && Math.abs(troughTrend) < 0.1) {
          const convergence = Math.abs(peakTrend - troughTrend);
          
          if (convergence > 0.05) {
            patterns.push({
              type: PatternType.TRIANGLE,
              confidence: 1 - convergence,
              startIndex: i,
              endIndex: i + windowSize - 1,
              parameters: { peakTrend, troughTrend, convergence },
            });
          }
        }
      }
    }
    
    return patterns;
  }

  /**
   * Detect cycles in the signal
   */
  private detectCycles(values: number[]): PatternMatch[] {
    const patterns: PatternMatch[] = [];
    const acf = this.calculateAutocorrelation(values, Math.floor(values.length / 2));
    
    // Find peaks in autocorrelation
    const peaks = this.findPeaks(acf);
    
    for (const peak of peaks.slice(1, 4)) { // Skip lag 0, check first 3 peaks
      if (acf[peak] > 0.5) {
        patterns.push({
          type: PatternType.CONSOLIDATION,
          confidence: acf[peak],
          startIndex: 0,
          endIndex: values.length - 1,
          parameters: { period: peak, strength: acf[peak] },
        });
      }
    }
    
    return patterns;
  }

  /**
   * Find peaks in the signal
   */
  private findPeaks(values: number[]): number[] {
    const peaks: number[] = [];
    
    for (let i = 1; i < values.length - 1; i++) {
      if (values[i] > values[i - 1] && values[i] > values[i + 1]) {
        peaks.push(i);
      }
    }
    
    return peaks;
  }

  /**
   * Find troughs in the signal
   */
  private findTroughs(values: number[]): number[] {
    const troughs: number[] = [];
    
    for (let i = 1; i < values.length - 1; i++) {
      if (values[i] < values[i - 1] && values[i] < values[i + 1]) {
        troughs.push(i);
      }
    }
    
    return troughs;
  }

  /**
   * Calculate trend slope
   */
  private calculateTrend(values: number[]): number {
    const x = Array.from({ length: values.length }, (_, i) => i);
    const regression = ss.linearRegression(x.map((xi, i) => [xi, values[i]]));
    return regression.m;
  }

  /**
   * Merge overlapping patterns
   */
  private mergeOverlappingPatterns(patterns: PatternMatch[]): PatternMatch[] {
    if (patterns.length === 0) return patterns;
    
    patterns.sort((a, b) => a.startIndex - b.startIndex);
    const merged: PatternMatch[] = [patterns[0]];
    
    for (let i = 1; i < patterns.length; i++) {
      const last = merged[merged.length - 1];
      const current = patterns[i];
      
      if (current.startIndex <= last.endIndex && current.type === last.type) {
        // Merge patterns
        last.endIndex = Math.max(last.endIndex, current.endIndex);
        last.confidence = Math.max(last.confidence, current.confidence);
      } else {
        merged.push(current);
      }
    }
    
    return merged;
  }

  /**
   * Calculate anomaly probability
   */
  private calculateAnomalyProbability(zscore: number): number {
    // Approximate normal CDF
    const t = 1 / (1 + 0.2316419 * Math.abs(zscore));
    const d = 0.3989423 * Math.exp(-zscore * zscore / 2);
    const probability = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    
    return zscore > 0 ? 1 - probability : probability;
  }

  /**
   * Classify anomaly type
   */
  private classifyAnomaly(values: number[], index: number): AnomalyPoint['type'] {
    const windowSize = 5;
    const start = Math.max(0, index - windowSize);
    const end = Math.min(values.length, index + windowSize);
    
    const before = values.slice(start, index);
    const after = values.slice(index + 1, end);
    
    const meanBefore = before.length > 0 ? ss.mean(before) : 0;
    const meanAfter = after.length > 0 ? ss.mean(after) : 0;
    const currentValue = values[index];
    
    if (currentValue > meanBefore && currentValue > meanAfter) {
      return 'spike';
    } else if (currentValue < meanBefore && currentValue < meanAfter) {
      return 'dip';
    } else if (Math.abs(meanAfter - meanBefore) > ss.standardDeviation(values)) {
      return 'level_shift';
    } else {
      return 'variance_change';
    }
  }

  /**
   * Isolation Forest anomaly detection
   */
  private isolationForest(values: number[], contamination: number = 0.1): AnomalyPoint[] {
    const anomalies: AnomalyPoint[] = [];
    const numTrees = 100;
    const sampleSize = Math.min(256, values.length);
    
    // Simplified isolation forest implementation
    const scores: number[] = new Array(values.length).fill(0);
    
    for (let tree = 0; tree < numTrees; tree++) {
      const sample = this.randomSample(values, sampleSize);
      const depths = this.calculateIsolationDepths(values, sample);
      
      for (let i = 0; i < values.length; i++) {
        scores[i] += depths[i];
      }
    }
    
    // Normalize scores
    const avgPathLength = 2 * (Math.log(sampleSize - 1) + 0.5772156649) - (2 * (sampleSize - 1) / sampleSize);
    const anomalyScores = scores.map(s => Math.pow(2, -s / (numTrees * avgPathLength)));
    
    // Find anomalies based on contamination factor
    const threshold = ss.quantile(anomalyScores, 1 - contamination);
    
    for (let i = 0; i < anomalyScores.length; i++) {
      if (anomalyScores[i] > threshold) {
        anomalies.push({
          index: i,
          value: values[i],
          zscore: (values[i] - ss.mean(values)) / ss.standardDeviation(values),
          probability: anomalyScores[i],
          type: this.classifyAnomaly(values, i),
        });
      }
    }
    
    return anomalies;
  }

  /**
   * Random sampling
   */
  private randomSample<T>(array: T[], size: number): T[] {
    const sample: T[] = [];
    const indices = new Set<number>();
    
    while (indices.size < size && indices.size < array.length) {
      indices.add(Math.floor(Math.random() * array.length));
    }
    
    indices.forEach(i => sample.push(array[i]));
    return sample;
  }

  /**
   * Calculate isolation depths
   */
  private calculateIsolationDepths(values: number[], sample: number[]): number[] {
    const depths: number[] = [];
    
    for (const value of values) {
      let depth = 0;
      let min = Math.min(...sample);
      let max = Math.max(...sample);
      
      while (min < max && depth < 10) {
        const split = min + Math.random() * (max - min);
        
        if (value < split) {
          max = split;
        } else {
          min = split;
        }
        
        depth++;
      }
      
      depths.push(depth);
    }
    
    return depths;
  }
}
