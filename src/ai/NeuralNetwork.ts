import * as tf from '@tensorflow/tfjs-node';
import { EventEmitter } from 'events';
import { Logger } from '@utils/logger';
import { MetricsCollector } from '@utils/metrics';
import { CacheManager } from '@utils/cache';

export interface NeuralNetworkConfig {
  inputShape: number[];
  hiddenLayers: number[];
  outputShape: number;
  learningRate: number;
  dropout: number;
  batchSize: number;
  epochs: number;
  validationSplit: number;
  earlyStopping: boolean;
  patience: number;
}

export interface PredictionResult {
  prediction: number[];
  confidence: number;
  timestamp: number;
  processingTime: number;
  modelVersion: string;
}

export interface TrainingMetrics {
  loss: number;
  accuracy: number;
  valLoss: number;
  valAccuracy: number;
  epoch: number;
}

/**
 * Advanced neural network for wallet signal prediction
 * Implements LSTM with attention mechanism for time series analysis
 */
export class NeuralNetwork extends EventEmitter {
  private model: tf.LayersModel | null = null;
  private readonly config: NeuralNetworkConfig;
  private readonly logger: Logger;
  private readonly metrics: MetricsCollector;
  private readonly cache: CacheManager;
  private isTraining: boolean = false;
  private modelVersion: string = '2.1.0';

  constructor(config: NeuralNetworkConfig) {
    super();
    this.config = config;
    this.logger = new Logger('NeuralNetwork');
    this.metrics = new MetricsCollector('ai.neural_network');
    this.cache = new CacheManager('neural_predictions');
  }

  /**
   * Initialize the neural network model
   */
  public async initialize(): Promise<void> {
    try {
      this.logger.info('Initializing neural network model');
      
      this.model = await this.buildModel();
      
      // Attempt to load pre-trained weights
      await this.loadWeights();
      
      this.emit('initialized', { modelVersion: this.modelVersion });
      this.logger.info('Neural network initialized successfully');
    } catch (error) {
      this.logger.error('Failed to initialize neural network', error);
      throw error;
    }
  }

  /**
   * Build the neural network architecture
   */
  private buildModel(): tf.LayersModel {
    const model = tf.sequential();

    // Input layer with LSTM
    model.add(tf.layers.lstm({
      units: this.config.hiddenLayers[0],
      returnSequences: true,
      inputShape: this.config.inputShape,
      dropout: this.config.dropout,
      recurrentDropout: this.config.dropout,
      kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }),
    }));

    // Attention mechanism
    model.add(tf.layers.multiHeadAttention({
      numHeads: 8,
      keyDim: 64,
      dropout: this.config.dropout,
    }));

    // Hidden LSTM layers
    for (let i = 1; i < this.config.hiddenLayers.length; i++) {
      const returnSequences = i < this.config.hiddenLayers.length - 1;
      
      model.add(tf.layers.lstm({
        units: this.config.hiddenLayers[i],
        returnSequences,
        dropout: this.config.dropout,
        recurrentDropout: this.config.dropout,
      }));

      // Batch normalization
      model.add(tf.layers.batchNormalization());
      
      // Add residual connections for deeper networks
      if (i % 2 === 0 && i > 0) {
        model.add(tf.layers.add());
      }
    }

    // Dense layers
    model.add(tf.layers.dense({
      units: 256,
      activation: 'relu',
      kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }),
    }));

    model.add(tf.layers.dropout({ rate: this.config.dropout }));

    model.add(tf.layers.dense({
      units: 128,
      activation: 'relu',
    }));

    // Output layer
    model.add(tf.layers.dense({
      units: this.config.outputShape,
      activation: 'sigmoid',
    }));

    // Compile the model
    model.compile({
      optimizer: tf.train.adam(this.config.learningRate),
      loss: 'binaryCrossentropy',
      metrics: ['accuracy', 'precision', 'recall'],
    });

    return model;
  }

  /**
   * Train the neural network with provided data
   */
  public async train(
    xTrain: tf.Tensor,
    yTrain: tf.Tensor,
    xVal?: tf.Tensor,
    yVal?: tf.Tensor,
  ): Promise<TrainingMetrics[]> {
    if (!this.model) {
      throw new Error('Model not initialized');
    }

    if (this.isTraining) {
      throw new Error('Training already in progress');
    }

    this.isTraining = true;
    const trainingMetrics: TrainingMetrics[] = [];

    try {
      this.logger.info('Starting neural network training');
      this.emit('training:start', { epochs: this.config.epochs });

      const callbacks: tf.CustomCallbackArgs = {
        onEpochEnd: async (epoch: number, logs?: tf.Logs) => {
          const metrics: TrainingMetrics = {
            loss: logs?.loss as number,
            accuracy: logs?.acc as number,
            valLoss: logs?.val_loss as number,
            valAccuracy: logs?.val_acc as number,
            epoch,
          };

          trainingMetrics.push(metrics);
          this.metrics.record('training.epoch', metrics);
          this.emit('training:epoch', metrics);

          // Early stopping logic
          if (this.config.earlyStopping && epoch > this.config.patience) {
            const recentLosses = trainingMetrics
              .slice(-this.config.patience)
              .map(m => m.valLoss);
            
            const isImproving = recentLosses[0] > recentLosses[recentLosses.length - 1];
            
            if (!isImproving) {
              this.logger.info('Early stopping triggered');
              return false; // Stop training
            }
          }
        },

        onBatchEnd: async (batch: number, logs?: tf.Logs) => {
          this.emit('training:batch', { batch, logs });
        },
      };

      const validationData = xVal && yVal ? [xVal, yVal] : undefined;

      await this.model.fit(xTrain, yTrain, {
        batchSize: this.config.batchSize,
        epochs: this.config.epochs,
        validationData,
        validationSplit: validationData ? 0 : this.config.validationSplit,
        callbacks,
        verbose: 0,
      });

      await this.saveWeights();
      this.updateModelVersion();

      this.logger.info('Neural network training completed');
      this.emit('training:complete', { metrics: trainingMetrics });

      return trainingMetrics;
    } catch (error) {
      this.logger.error('Training failed', error);
      this.emit('training:error', error);
      throw error;
    } finally {
      this.isTraining = false;
    }
  }

  /**
   * Make predictions using the trained model
   */
  public async predict(input: tf.Tensor | number[][]): Promise<PredictionResult> {
    if (!this.model) {
      throw new Error('Model not initialized');
    }

    const startTime = Date.now();
    
    try {
      // Check cache first
      const cacheKey = this.generateCacheKey(input);
      const cached = await this.cache.get(cacheKey);
      
      if (cached) {
        this.metrics.increment('predictions.cache_hit');
        return cached as PredictionResult;
      }

      // Convert input to tensor if necessary
      const inputTensor = input instanceof tf.Tensor 
        ? input 
        : tf.tensor(input);

      // Normalize input
      const normalized = this.normalizeInput(inputTensor);

      // Make prediction
      const output = this.model.predict(normalized) as tf.Tensor;
      const prediction = await output.array() as number[];

      // Calculate confidence
      const confidence = this.calculateConfidence(prediction);

      const result: PredictionResult = {
        prediction,
        confidence,
        timestamp: Date.now(),
        processingTime: Date.now() - startTime,
        modelVersion: this.modelVersion,
      };

      // Cache the result
      await this.cache.set(cacheKey, result, 300); // Cache for 5 minutes

      // Record metrics
      this.metrics.record('predictions.processing_time', result.processingTime);
      this.metrics.record('predictions.confidence', confidence);
      this.metrics.increment('predictions.total');

      this.emit('prediction', result);

      // Cleanup tensors
      if (!(input instanceof tf.Tensor)) {
        inputTensor.dispose();
      }
      normalized.dispose();
      output.dispose();

      return result;
    } catch (error) {
      this.logger.error('Prediction failed', error);
      this.metrics.increment('predictions.errors');
      throw error;
    }
  }

  /**
   * Evaluate model performance
   */
  public async evaluate(xTest: tf.Tensor, yTest: tf.Tensor): Promise<{
    loss: number;
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
  }> {
    if (!this.model) {
      throw new Error('Model not initialized');
    }

    try {
      const evaluation = await this.model.evaluate(xTest, yTest) as tf.Scalar[];
      
      const loss = await evaluation[0].data();
      const accuracy = await evaluation[1].data();
      const precision = await evaluation[2].data();
      const recall = await evaluation[3].data();

      const f1Score = 2 * (precision[0] * recall[0]) / (precision[0] + recall[0]);

      const metrics = {
        loss: loss[0],
        accuracy: accuracy[0],
        precision: precision[0],
        recall: recall[0],
        f1Score,
      };

      this.logger.info('Model evaluation completed', metrics);
      this.metrics.record('evaluation', metrics);

      return metrics;
    } catch (error) {
      this.logger.error('Evaluation failed', error);
      throw error;
    }
  }

  /**
   * Perform incremental learning with new data
   */
  public async incrementalLearning(
    newData: tf.Tensor,
    newLabels: tf.Tensor,
    learningRate?: number,
  ): Promise<void> {
    if (!this.model) {
      throw new Error('Model not initialized');
    }

    try {
      this.logger.info('Starting incremental learning');

      // Adjust learning rate for fine-tuning
      const lr = learningRate || this.config.learningRate * 0.1;
      
      this.model.compile({
        optimizer: tf.train.adam(lr),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy'],
      });

      await this.model.fit(newData, newLabels, {
        batchSize: Math.min(32, newData.shape[0]),
        epochs: 10,
        verbose: 0,
      });

      await this.saveWeights();
      this.logger.info('Incremental learning completed');
    } catch (error) {
      this.logger.error('Incremental learning failed', error);
      throw error;
    }
  }

  /**
   * Normalize input data
   */
  private normalizeInput(input: tf.Tensor): tf.Tensor {
    // Z-score normalization
    const mean = input.mean();
    const std = input.sub(mean).square().mean().sqrt();
    return input.sub(mean).div(std.add(1e-8));
  }

  /**
   * Calculate prediction confidence
   */
  private calculateConfidence(prediction: number[]): number {
    const max = Math.max(...prediction);
    const sum = prediction.reduce((a, b) => a + b, 0);
    const entropy = -prediction
      .map(p => p === 0 ? 0 : p * Math.log2(p))
      .reduce((a, b) => a + b, 0);
    
    // Combine max probability with entropy-based confidence
    const confidence = (max / sum) * (1 - entropy / Math.log2(prediction.length));
    
    return Math.min(Math.max(confidence, 0), 1);
  }

  /**
   * Generate cache key for predictions
   */
  private generateCacheKey(input: tf.Tensor | number[][]): string {
    const data = input instanceof tf.Tensor 
      ? input.dataSync() 
      : input.flat();
    
    return `prediction_${this.modelVersion}_${data.slice(0, 10).join('_')}`;
  }

  /**
   * Save model weights
   */
  private async saveWeights(): Promise<void> {
    if (!this.model) return;

    try {
      await this.model.save('file://./models/neural_network');
      this.logger.info('Model weights saved successfully');
    } catch (error) {
      this.logger.error('Failed to save model weights', error);
    }
  }

  /**
   * Load pre-trained model weights
   */
  private async loadWeights(): Promise<void> {
    try {
      this.model = await tf.loadLayersModel('file://./models/neural_network/model.json');
      this.logger.info('Pre-trained weights loaded successfully');
    } catch (error) {
      this.logger.warn('No pre-trained weights found, using random initialization');
    }
  }

  /**
   * Update model version after training
   */
  private updateModelVersion(): void {
    const [major, minor, patch] = this.modelVersion.split('.').map(Number);
    this.modelVersion = `${major}.${minor}.${patch + 1}`;
  }

  /**
   * Dispose of the model and free resources
   */
  public dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    this.removeAllListeners();
    this.logger.info('Neural network disposed');
  }
}
