import { WalletTracker } from '../src/blockchain/WalletTracker';
import { NeuralNetwork } from '../src/ai/NeuralNetwork';
import { SignalProcessor } from '../src/ai/SignalProcessor';

async function example() {
  // Initialize wallet tracker
  const tracker = new WalletTracker(
    'https://api.mainnet-beta.solana.com',
    'wss://api.mainnet-beta.solana.com'
  );
  
  await tracker.initialize();
  
  // Track a wallet
  const wallet = await tracker.trackWallet(
    'DYw8jCTfwHNRJhhmFcbXvVDTqWMEVFBX6ZKUmG5CNSKK'
  );
  
  // Get metrics
  const metrics = await tracker.getWalletMetrics(wallet.address);
  console.log('Wallet Metrics:', metrics);
  
  // Analyze wallet
  const analysis = await tracker.analyzeWallet(wallet.address);
  console.log('Analysis:', analysis);
}

example().catch(console.error);
