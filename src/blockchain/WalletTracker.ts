import { Connection, PublicKey, ParsedTransactionWithMeta, ConfirmedSignatureInfo } from '@solana/web3.js';
import { EventEmitter } from 'events';
import { Logger } from '@utils/logger';
import { MetricsCollector } from '@utils/metrics';
import { CacheManager } from '@utils/cache';
import { SignalProcessor } from '@ai/SignalProcessor';
import { NeuralNetwork } from '@ai/NeuralNetwork';

export interface WalletActivity {
  address: string;
  timestamp: number;
  type: ActivityType;
  signature: string;
  amount?: number;
  token?: string;
  counterparty?: string;
  fee?: number;
  status: TransactionStatus;
  metadata?: Record<string, unknown>;
}

export interface WalletMetrics {
  totalTransactions: number;
  volume24h: number;
  uniqueTokens: number;
  avgTransactionSize: number;
  transactionFrequency: number;
  riskScore: number;
  profitLoss: number;
  winRate: number;
}

export interface TrackedWallet {
  address: string;
  label?: string;
  startTime: number;
  lastActivity: number;
  metrics: WalletMetrics;
  activities: WalletActivity[];
  signals: SignalData[];
  predictions: PredictionData[];
}

export interface SignalData {
  timestamp: number;
  strength: number;
  type: string;
  confidence: number;
}

export interface PredictionData {
  timestamp: number;
  prediction: number[];
  confidence: number;
  actualOutcome?: number;
}

export enum ActivityType {
  TRANSFER = 'transfer',
  SWAP = 'swap',
  STAKE = 'stake',
  UNSTAKE = 'unstake',
  MINT = 'mint',
  BURN = 'burn',
  CREATE_ACCOUNT = 'create_account',
  CLOSE_ACCOUNT = 'close_account',
  UNKNOWN = 'unknown',
}

export enum TransactionStatus {
  SUCCESS = 'success',
  FAILED = 'failed',
  PENDING = 'pending',
}

/**
 * Advanced wallet tracking system for Solana blockchain
 */
export class WalletTracker extends EventEmitter {
  private readonly connection: Connection;
  private readonly logger: Logger;
  private readonly metrics: MetricsCollector;
  private readonly cache: CacheManager;
  private readonly signalProcessor: SignalProcessor;
  private readonly neuralNetwork: NeuralNetwork;
  private readonly trackedWallets: Map<string, TrackedWallet>;
  private readonly subscriptions: Map<string, number>;
  private isRunning: boolean = false;
  private pollInterval: NodeJS.Timeout | null = null;

  constructor(
    rpcEndpoint: string,
    wsEndpoint?: string,
    commitment: 'processed' | 'confirmed' | 'finalized' = 'confirmed',
  ) {
    super();
    
    this.connection = new Connection(rpcEndpoint, {
      commitment,
      wsEndpoint,
    });
    
    this.logger = new Logger('WalletTracker');
    this.metrics = new MetricsCollector('blockchain.wallet_tracker');
    this.cache = new CacheManager('wallet_data');
    this.signalProcessor = new SignalProcessor();
    this.neuralNetwork = new NeuralNetwork({
      inputShape: [60, 10],
      hiddenLayers: [256, 128, 64],
      outputShape: 3,
      learningRate: 0.001,
      dropout: 0.2,
      batchSize: 32,
      epochs: 100,
      validationSplit: 0.2,
      earlyStopping: true,
      patience: 10,
    });
    
    this.trackedWallets = new Map();
    this.subscriptions = new Map();
  }

  /**
   * Initialize the wallet tracker
   */
  public async initialize(): Promise<void> {
    try {
      this.logger.info('Initializing wallet tracker');
      
      // Initialize AI components
      await this.neuralNetwork.initialize();
      
      // Test connection
      const version = await this.connection.getVersion();
      this.logger.info(`Connected to Solana network: ${version['solana-core']}`);
      
      // Start monitoring
      await this.startMonitoring();
      
      this.emit('initialized');
      this.logger.info('Wallet tracker initialized successfully');
    } catch (error) {
      this.logger.error('Failed to initialize wallet tracker', error);
      throw error;
    }
  }

  /**
   * Track a new wallet
   */
  public async trackWallet(address: string, label?: string): Promise<TrackedWallet> {
    try {
      const publicKey = new PublicKey(address);
      
      // Validate address
      const accountInfo = await this.connection.getAccountInfo(publicKey);
      if (!accountInfo) {
        throw new Error('Invalid or non-existent wallet address');
      }
      
      // Check if already tracking
      if (this.trackedWallets.has(address)) {
        this.logger.warn(`Already tracking wallet: ${address}`);
        return this.trackedWallets.get(address)!;
      }
      
      // Create tracked wallet entry
      const trackedWallet: TrackedWallet = {
        address,
        label,
        startTime: Date.now(),
        lastActivity: Date.now(),
        metrics: this.initializeMetrics(),
        activities: [],
        signals: [],
        predictions: [],
      };
      
      this.trackedWallets.set(address, trackedWallet);
      
      // Subscribe to account changes
      await this.subscribeToWallet(address);
      
      // Load historical data
      await this.loadHistoricalData(address);
      
      this.logger.info(`Started tracking wallet: ${address}`);
      this.metrics.increment('wallets.tracked');
      this.emit('wallet:tracked', trackedWallet);
      
      return trackedWallet;
    } catch (error) {
      this.logger.error(`Failed to track wallet: ${address}`, error);
      throw error;
    }
  }

  /**
   * Stop tracking a wallet
   */
  public async untrackWallet(address: string): Promise<void> {
    try {
      if (!this.trackedWallets.has(address)) {
        throw new Error('Wallet not being tracked');
      }
      
      // Unsubscribe from updates
      await this.unsubscribeFromWallet(address);
      
      // Remove from tracked wallets
      this.trackedWallets.delete(address);
      
      this.logger.info(`Stopped tracking wallet: ${address}`);
      this.metrics.decrement('wallets.tracked');
      this.emit('wallet:untracked', address);
    } catch (error) {
      this.logger.error(`Failed to untrack wallet: ${address}`, error);
      throw error;
    }
  }

  /**
   * Get wallet metrics
   */
  public async getWalletMetrics(address: string): Promise<WalletMetrics> {
    const wallet = this.trackedWallets.get(address);
    if (!wallet) {
      throw new Error('Wallet not being tracked');
    }
    
    // Update metrics
    await this.updateWalletMetrics(wallet);
    
    return wallet.metrics;
  }

  /**
   * Get wallet activities
   */
  public getWalletActivities(
    address: string,
    limit?: number,
    offset?: number,
  ): WalletActivity[] {
    const wallet = this.trackedWallets.get(address);
    if (!wallet) {
      throw new Error('Wallet not being tracked');
    }
    
    const start = offset || 0;
    const end = limit ? start + limit : undefined;
    
    return wallet.activities.slice(start, end);
  }

  /**
   * Analyze wallet behavior
   */
  public async analyzeWallet(address: string): Promise<{
    patterns: PatternMatch[];
    anomalies: AnomalyPoint[];
    predictions: PredictionResult;
    riskAssessment: RiskAssessment;
  }> {
    const wallet = this.trackedWallets.get(address);
    if (!wallet) {
      throw new Error('Wallet not being tracked');
    }
    
    try {
      // Process signals
      const signalData = wallet.activities.map(a => ({
        timestamp: a.timestamp,
        value: a.amount || 0,
        volume: a.amount,
      }));
      
      const processedSignal = await this.signalProcessor.processSignal(signalData);
      
      // Generate predictions
      const inputData = this.prepareInputData(wallet);
      const predictions = await this.neuralNetwork.predict(inputData);
      
      // Assess risk
      const riskAssessment = await this.assessRisk(wallet, processedSignal);
      
      // Store analysis results
      wallet.signals.push({
        timestamp: Date.now(),
        strength: processedSignal.features.mean,
        type: 'analysis',
        confidence: predictions.confidence,
      });
      
      wallet.predictions.push({
        timestamp: Date.now(),
        prediction: predictions.prediction,
        confidence: predictions.confidence,
      });
      
      return {
        patterns: processedSignal.patterns,
        anomalies: processedSignal.anomalies,
        predictions,
        riskAssessment,
      };
    } catch (error) {
      this.logger.error(`Failed to analyze wallet: ${address}`, error);
      throw error;
    }
  }

  /**
   * Start monitoring wallets
   */
  private async startMonitoring(): Promise<void> {
    if (this.isRunning) return;
    
    this.isRunning = true;
    
    // Set up polling interval
    this.pollInterval = setInterval(async () => {
      await this.pollWalletUpdates();
    }, 5000); // Poll every 5 seconds
    
    this.logger.info('Started wallet monitoring');
  }

  /**
   * Stop monitoring wallets
   */
  private async stopMonitoring(): Promise<void> {
    if (!this.isRunning) return;
    
    this.isRunning = false;
    
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
    
    this.logger.info('Stopped wallet monitoring');
  }

  /**
   * Poll for wallet updates
   */
  private async pollWalletUpdates(): Promise<void> {
    const promises: Promise<void>[] = [];
    
    for (const [address, wallet] of this.trackedWallets) {
      promises.push(this.updateWalletActivity(wallet));
    }
    
    await Promise.allSettled(promises);
  }

  /**
   * Update wallet activity
   */
  private async updateWalletActivity(wallet: TrackedWallet): Promise<void> {
    try {
      const publicKey = new PublicKey(wallet.address);
      
      // Get recent signatures
      const signatures = await this.connection.getSignaturesForAddress(
        publicKey,
        { limit: 10 },
      );
      
      for (const sig of signatures) {
        // Check if we've already processed this transaction
        if (wallet.activities.some(a => a.signature === sig.signature)) {
          continue;
        }
        
        // Get transaction details
        const tx = await this.getTransactionDetails(sig.signature);
        if (!tx) continue;
        
        // Parse activity
        const activity = await this.parseTransaction(tx, wallet.address);
        if (activity) {
          wallet.activities.push(activity);
          wallet.lastActivity = activity.timestamp;
          
          // Emit activity event
          this.emit('wallet:activity', { wallet: wallet.address, activity });
          
          // Update metrics
          await this.updateWalletMetrics(wallet);
          
          // Check for signals
          await this.checkForSignals(wallet, activity);
        }
      }
    } catch (error) {
      this.logger.error(`Failed to update wallet activity: ${wallet.address}`, error);
    }
  }

  /**
   * Subscribe to wallet updates
   */
  private async subscribeToWallet(address: string): Promise<void> {
    try {
      const publicKey = new PublicKey(address);
      
      // Subscribe to account changes
      const subscriptionId = this.connection.onAccountChange(
        publicKey,
        (accountInfo, context) => {
          this.handleAccountChange(address, accountInfo, context);
        },
        'confirmed',
      );
      
      this.subscriptions.set(address, subscriptionId);
      
      // Subscribe to logs
      const logSubscriptionId = this.connection.onLogs(
        publicKey,
        (logs, context) => {
          this.handleLogs(address, logs, context);
        },
        'confirmed',
      );
      
      this.subscriptions.set(`${address}_logs`, logSubscriptionId);
      
      this.logger.debug(`Subscribed to wallet updates: ${address}`);
    } catch (error) {
      this.logger.error(`Failed to subscribe to wallet: ${address}`, error);
      throw error;
    }
  }

  /**
   * Unsubscribe from wallet updates
   */
  private async unsubscribeFromWallet(address: string): Promise<void> {
    try {
      // Remove account subscription
      const subscriptionId = this.subscriptions.get(address);
      if (subscriptionId !== undefined) {
        await this.connection.removeAccountChangeListener(subscriptionId);
        this.subscriptions.delete(address);
      }
      
      // Remove log subscription
      const logSubscriptionId = this.subscriptions.get(`${address}_logs`);
      if (logSubscriptionId !== undefined) {
        await this.connection.removeOnLogsListener(logSubscriptionId);
        this.subscriptions.delete(`${address}_logs`);
      }
      
      this.logger.debug(`Unsubscribed from wallet updates: ${address}`);
    } catch (error) {
      this.logger.error(`Failed to unsubscribe from wallet: ${address}`, error);
      throw error;
    }
  }

  /**
   * Handle account changes
   */
  private handleAccountChange(
    address: string,
    accountInfo: any,
    context: any,
  ): void {
    try {
      const wallet = this.trackedWallets.get(address);
      if (!wallet) return;
      
      this.logger.debug(`Account change detected for ${address}`);
      
      // Update wallet activity
      this.updateWalletActivity(wallet).catch(error => {
        this.logger.error('Failed to update wallet on account change', error);
      });
      
      this.emit('wallet:changed', { address, accountInfo, context });
    } catch (error) {
      this.logger.error('Error handling account change', error);
    }
  }

  /**
   * Handle transaction logs
   */
  private handleLogs(address: string, logs: any, context: any): void {
    try {
      const wallet = this.trackedWallets.get(address);
      if (!wallet) return;
      
      this.logger.debug(`Logs received for ${address}`);
      
      // Parse logs for relevant information
      if (logs.err) {
        this.logger.warn(`Transaction error for ${address}: ${logs.err}`);
        this.metrics.increment('transactions.errors');
      }
      
      this.emit('wallet:logs', { address, logs, context });
    } catch (error) {
      this.logger.error('Error handling logs', error);
    }
  }

  /**
   * Load historical data for a wallet
   */
  private async loadHistoricalData(address: string): Promise<void> {
    try {
      const wallet = this.trackedWallets.get(address);
      if (!wallet) return;
      
      const publicKey = new PublicKey(address);
      
      // Get historical signatures
      const signatures = await this.connection.getSignaturesForAddress(
        publicKey,
        { limit: 100 },
      );
      
      // Process transactions in batches
      const batchSize = 10;
      for (let i = 0; i < signatures.length; i += batchSize) {
        const batch = signatures.slice(i, i + batchSize);
        const promises = batch.map(sig => this.getTransactionDetails(sig.signature));
        const transactions = await Promise.all(promises);
        
        for (const tx of transactions) {
          if (!tx) continue;
          
          const activity = await this.parseTransaction(tx, address);
          if (activity) {
            wallet.activities.push(activity);
          }
        }
      }
      
      // Sort activities by timestamp
      wallet.activities.sort((a, b) => a.timestamp - b.timestamp);
      
      this.logger.info(`Loaded ${wallet.activities.length} historical activities for ${address}`);
    } catch (error) {
      this.logger.error(`Failed to load historical data for ${address}`, error);
    }
  }

  /**
   * Get transaction details
   */
  private async getTransactionDetails(signature: string): Promise<ParsedTransactionWithMeta | null> {
    try {
      // Check cache first
      const cached = await this.cache.get(`tx_${signature}`);
      if (cached) {
        return cached as ParsedTransactionWithMeta;
      }
      
      const tx = await this.connection.getParsedTransaction(signature, {
        maxSupportedTransactionVersion: 0,
      });
      
      if (tx) {
        // Cache transaction
        await this.cache.set(`tx_${signature}`, tx, 3600); // Cache for 1 hour
      }
      
      return tx;
    } catch (error) {
      this.logger.error(`Failed to get transaction details: ${signature}`, error);
      return null;
    }
  }

  /**
   * Parse transaction into activity
   */
  private async parseTransaction(
    tx: ParsedTransactionWithMeta,
    walletAddress: string,
  ): Promise<WalletActivity | null> {
    try {
      if (!tx.meta || !tx.transaction) return null;
      
      const activity: WalletActivity = {
        address: walletAddress,
        timestamp: (tx.blockTime || 0) * 1000,
        type: this.determineActivityType(tx),
        signature: tx.transaction.signatures[0],
        status: tx.meta.err ? TransactionStatus.FAILED : TransactionStatus.SUCCESS,
        fee: tx.meta.fee / 1e9, // Convert lamports to SOL
        metadata: {
          slot: tx.slot,
          computeUnitsConsumed: tx.meta.computeUnitsConsumed,
        },
      };
      
      // Extract transfer amount and counterparty
      const transferInfo = this.extractTransferInfo(tx, walletAddress);
      if (transferInfo) {
        activity.amount = transferInfo.amount;
        activity.token = transferInfo.token;
        activity.counterparty = transferInfo.counterparty;
      }
      
      return activity;
    } catch (error) {
      this.logger.error('Failed to parse transaction', error);
      return null;
    }
  }

  /**
   * Determine activity type from transaction
   */
  private determineActivityType(tx: ParsedTransactionWithMeta): ActivityType {
    if (!tx.transaction.message.instructions) return ActivityType.UNKNOWN;
    
    for (const instruction of tx.transaction.message.instructions) {
      if ('parsed' in instruction) {
        const { type } = instruction.parsed;
        
        switch (type) {
          case 'transfer':
          case 'transferChecked':
            return ActivityType.TRANSFER;
          case 'create':
          case 'createAccount':
            return ActivityType.CREATE_ACCOUNT;
          case 'closeAccount':
            return ActivityType.CLOSE_ACCOUNT;
          case 'mintTo':
            return ActivityType.MINT;
          case 'burn':
            return ActivityType.BURN;
          default:
            // Check for swap programs
            if (this.isSwapProgram(instruction.programId.toString())) {
              return ActivityType.SWAP;
            }
        }
      }
    }
    
    return ActivityType.UNKNOWN;
  }

  /**
   * Check if program is a known swap program
   */
  private isSwapProgram(programId: string): boolean {
    const swapPrograms = [
      'JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB', // Jupiter
      '9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP', // Orca
      'whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc', // Whirlpool
      'SSwpkEEcbUqx4vtoEByFjSkhKdCT862DNVb52nZg1UZ', // Saber
    ];
    
    return swapPrograms.includes(programId);
  }

  /**
   * Extract transfer information from transaction
   */
  private extractTransferInfo(
    tx: ParsedTransactionWithMeta,
    walletAddress: string,
  ): { amount: number; token: string; counterparty: string } | null {
    if (!tx.meta) return null;
    
    // Check pre and post balances for SOL transfers
    const accountKeys = tx.transaction.message.accountKeys;
    const walletIndex = accountKeys.findIndex(
      key => key.pubkey.toString() === walletAddress
    );
    
    if (walletIndex !== -1) {
      const preBalance = tx.meta.preBalances[walletIndex];
      const postBalance = tx.meta.postBalances[walletIndex];
      const change = (postBalance - preBalance) / 1e9; // Convert to SOL
      
      if (Math.abs(change) > 0.0001) {
        // Find counterparty
        let counterparty = '';
        for (let i = 0; i < accountKeys.length; i++) {
          if (i !== walletIndex) {
            const preB = tx.meta.preBalances[i];
            const postB = tx.meta.postBalances[i];
            if (Math.abs(postB - preB) > 0) {
              counterparty = accountKeys[i].pubkey.toString();
              break;
            }
          }
        }
        
        return {
          amount: Math.abs(change),
          token: 'SOL',
          counterparty,
        };
      }
    }
    
    // Check for token transfers
    if (tx.meta.preTokenBalances && tx.meta.postTokenBalances) {
      for (let i = 0; i < tx.meta.preTokenBalances.length; i++) {
        const pre = tx.meta.preTokenBalances[i];
        const post = tx.meta.postTokenBalances[i];
        
        if (pre.owner === walletAddress || post.owner === walletAddress) {
          const preAmount = parseFloat(pre.uiTokenAmount.uiAmountString || '0');
          const postAmount = parseFloat(post.uiTokenAmount.uiAmountString || '0');
          const change = postAmount - preAmount;
          
          if (Math.abs(change) > 0) {
            return {
              amount: Math.abs(change),
              token: pre.mint,
              counterparty: pre.owner === walletAddress ? post.owner : pre.owner,
            };
          }
        }
      }
    }
    
    return null;
  }

  /**
   * Initialize wallet metrics
   */
  private initializeMetrics(): WalletMetrics {
    return {
      totalTransactions: 0,
      volume24h: 0,
      uniqueTokens: 0,
      avgTransactionSize: 0,
      transactionFrequency: 0,
      riskScore: 0,
      profitLoss: 0,
      winRate: 0,
    };
  }

  /**
   * Update wallet metrics
   */
  private async updateWalletMetrics(wallet: TrackedWallet): Promise<void> {
    const now = Date.now();
    const oneDayAgo = now - 24 * 60 * 60 * 1000;
    
    // Calculate 24h volume
    const recentActivities = wallet.activities.filter(a => a.timestamp > oneDayAgo);
    wallet.metrics.volume24h = recentActivities.reduce((sum, a) => sum + (a.amount || 0), 0);
    
    // Total transactions
    wallet.metrics.totalTransactions = wallet.activities.length;
    
    // Unique tokens
    const tokens = new Set(wallet.activities.map(a => a.token).filter(Boolean));
    wallet.metrics.uniqueTokens = tokens.size;
    
    // Average transaction size
    const amounts = wallet.activities.map(a => a.amount || 0).filter(a => a > 0);
    wallet.metrics.avgTransactionSize = amounts.length > 0
      ? amounts.reduce((sum, a) => sum + a, 0) / amounts.length
      : 0;
    
    // Transaction frequency (transactions per day)
    const timeRange = now - wallet.startTime;
    const days = timeRange / (24 * 60 * 60 * 1000);
    wallet.metrics.transactionFrequency = wallet.activities.length / Math.max(days, 1);
    
    // Calculate P&L and win rate
    await this.calculateProfitMetrics(wallet);
    
    // Update risk score
    wallet.metrics.riskScore = await this.calculateRiskScore(wallet);
    
    this.metrics.record('wallet.metrics', wallet.metrics);
  }

  /**
   * Calculate profit metrics
   */
  private async calculateProfitMetrics(wallet: TrackedWallet): Promise<void> {
    // Simplified P&L calculation
    let totalIn = 0;
    let totalOut = 0;
    let wins = 0;
    let losses = 0;
    
    for (const activity of wallet.activities) {
      if (activity.type === ActivityType.SWAP && activity.amount) {
        // Track swaps for P&L
        // This is a simplified calculation
        if (activity.amount > 0) {
          totalIn += activity.amount;
          wins++;
        } else {
          totalOut += Math.abs(activity.amount);
          losses++;
        }
      }
    }
    
    wallet.metrics.profitLoss = totalIn - totalOut;
    wallet.metrics.winRate = wins + losses > 0 ? wins / (wins + losses) : 0;
  }

  /**
   * Calculate risk score for wallet
   */
  private async calculateRiskScore(wallet: TrackedWallet): Promise<number> {
    let riskScore = 0;
    
    // High transaction frequency increases risk
    if (wallet.metrics.transactionFrequency > 100) {
      riskScore += 0.2;
    }
    
    // Large volume increases risk
    if (wallet.metrics.volume24h > 10000) {
      riskScore += 0.2;
    }
    
    // Many unique tokens increases risk
    if (wallet.metrics.uniqueTokens > 50) {
      riskScore += 0.1;
    }
    
    // Failed transactions increase risk
    const failedTxs = wallet.activities.filter(a => a.status === TransactionStatus.FAILED);
    if (failedTxs.length / wallet.activities.length > 0.1) {
      riskScore += 0.3;
    }
    
    // New wallet increases risk
    const walletAge = Date.now() - wallet.startTime;
    if (walletAge < 7 * 24 * 60 * 60 * 1000) { // Less than 7 days
      riskScore += 0.2;
    }
    
    return Math.min(riskScore, 1);
  }

  /**
   * Check for trading signals
   */
  private async checkForSignals(wallet: TrackedWallet, activity: WalletActivity): Promise<void> {
    // Volume spike signal
    if (activity.amount && activity.amount > wallet.metrics.avgTransactionSize * 3) {
      wallet.signals.push({
        timestamp: Date.now(),
        strength: 0.8,
        type: 'volume_spike',
        confidence: 0.9,
      });
      
      this.emit('signal:detected', {
        wallet: wallet.address,
        signal: 'volume_spike',
        activity,
      });
    }
    
    // Rapid trading signal
    const recentActivities = wallet.activities.slice(-10);
    const timeSpan = recentActivities[recentActivities.length - 1].timestamp - recentActivities[0].timestamp;
    
    if (recentActivities.length === 10 && timeSpan < 60000) { // 10 txs in 1 minute
      wallet.signals.push({
        timestamp: Date.now(),
        strength: 0.9,
        type: 'rapid_trading',
        confidence: 0.85,
      });
      
      this.emit('signal:detected', {
        wallet: wallet.address,
        signal: 'rapid_trading',
        activity,
      });
    }
  }

  /**
   * Prepare input data for neural network
   */
  private prepareInputData(wallet: TrackedWallet): number[][] {
    const features: number[][] = [];
    const windowSize = 60;
    
    // Create time series features
    for (let i = 0; i < Math.min(wallet.activities.length, windowSize); i++) {
      const activity = wallet.activities[wallet.activities.length - 1 - i];
      
      features.push([
        activity.amount || 0,
        activity.fee || 0,
        activity.type === ActivityType.SWAP ? 1 : 0,
        activity.type === ActivityType.TRANSFER ? 1 : 0,
        activity.status === TransactionStatus.SUCCESS ? 1 : 0,
        wallet.metrics.volume24h,
        wallet.metrics.transactionFrequency,
        wallet.metrics.riskScore,
        wallet.metrics.profitLoss,
        wallet.metrics.winRate,
      ]);
    }
    
    // Pad with zeros if needed
    while (features.length < windowSize) {
      features.push(new Array(10).fill(0));
    }
    
    return features;
  }

  /**
   * Assess risk for wallet
   */
  private async assessRisk(
    wallet: TrackedWallet,
    signal: ProcessedSignal,
  ): Promise<RiskAssessment> {
    const riskFactors: RiskFactor[] = [];
    
    // Check for anomalies
    if (signal.anomalies.length > 0) {
      riskFactors.push({
        type: 'anomaly',
        severity: 'high',
        description: `Detected ${signal.anomalies.length} anomalies in trading pattern`,
        score: 0.8,
      });
    }
    
    // Check patterns
    const riskyPatterns = signal.patterns.filter(
      p => p.type === PatternType.REVERSAL || p.type === PatternType.BREAKOUT
    );
    
    if (riskyPatterns.length > 0) {
      riskFactors.push({
        type: 'pattern',
        severity: 'medium',
        description: `Detected risky patterns: ${riskyPatterns.map(p => p.type).join(', ')}`,
        score: 0.6,
      });
    }
    
    // Check metrics
    if (wallet.metrics.riskScore > 0.7) {
      riskFactors.push({
        type: 'metrics',
        severity: 'high',
        description: 'High risk score based on wallet metrics',
        score: wallet.metrics.riskScore,
      });
    }
    
    // Calculate overall risk
    const overallRisk = riskFactors.length > 0
      ? riskFactors.reduce((sum, f) => sum + f.score, 0) / riskFactors.length
      : 0;
    
    return {
      score: overallRisk,
      level: this.getRiskLevel(overallRisk),
      factors: riskFactors,
      recommendations: this.generateRiskRecommendations(riskFactors),
    };
  }

  /**
   * Get risk level from score
   */
  private getRiskLevel(score: number): 'low' | 'medium' | 'high' | 'critical' {
    if (score < 0.25) return 'low';
    if (score < 0.5) return 'medium';
    if (score < 0.75) return 'high';
    return 'critical';
  }

  /**
   * Generate risk recommendations
   */
  private generateRiskRecommendations(factors: RiskFactor[]): string[] {
    const recommendations: string[] = [];
    
    for (const factor of factors) {
      switch (factor.type) {
        case 'anomaly':
          recommendations.push('Monitor wallet closely for unusual activity');
          recommendations.push('Consider reducing exposure to this wallet');
          break;
        case 'pattern':
          recommendations.push('Wait for pattern confirmation before taking action');
          recommendations.push('Set stop-loss orders to limit potential losses');
          break;
        case 'metrics':
          recommendations.push('Review wallet history for suspicious behavior');
          recommendations.push('Consider diversifying tracking to multiple wallets');
          break;
      }
    }
    
    return [...new Set(recommendations)]; // Remove duplicates
  }

  /**
   * Cleanup and dispose resources
   */
  public async dispose(): Promise<void> {
    try {
      // Stop monitoring
      await this.stopMonitoring();
      
      // Unsubscribe from all wallets
      for (const address of this.trackedWallets.keys()) {
        await this.unsubscribeFromWallet(address);
      }
      
      // Clear data
      this.trackedWallets.clear();
      this.subscriptions.clear();
      
      // Dispose AI components
      this.neuralNetwork.dispose();
      
      this.logger.info('Wallet tracker disposed');
    } catch (error) {
      this.logger.error('Error disposing wallet tracker', error);
    }
  }
}

// Type definitions for risk assessment
interface RiskAssessment {
  score: number;
  level: 'low' | 'medium' | 'high' | 'critical';
  factors: RiskFactor[];
  recommendations: string[];
}

interface RiskFactor {
  type: 'anomaly' | 'pattern' | 'metrics';
  severity: 'low' | 'medium' | 'high';
  description: string;
  score: number;
}

// Re-export types from signal processor
import { PatternMatch, AnomalyPoint, PatternType, ProcessedSignal } from '@ai/SignalProcessor';
import { PredictionResult } from '@ai/NeuralNetwork';

export { PatternMatch, AnomalyPoint, PatternType, PredictionResult };
