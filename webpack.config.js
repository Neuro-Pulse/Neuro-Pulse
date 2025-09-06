const path = require('path');
const nodeExternals = require('webpack-node-externals');

module.exports = {
  mode: process.env.NODE_ENV || 'development',
  target: 'node',
  entry: './src/index.ts',
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'bundle.js',
  },
  resolve: {
    extensions: ['.ts', '.js'],
    alias: {
      '@': path.resolve(__dirname, 'src'),
      '@ai': path.resolve(__dirname, 'src/ai'),
      '@blockchain': path.resolve(__dirname, 'src/blockchain'),
      '@config': path.resolve(__dirname, 'src/config'),
      '@controllers': path.resolve(__dirname, 'src/controllers'),
      '@core': path.resolve(__dirname, 'src/core'),
      '@middleware': path.resolve(__dirname, 'src/middleware'),
      '@services': path.resolve(__dirname, 'src/services'),
      '@utils': path.resolve(__dirname, 'src/utils'),
    },
  },
  module: {
    rules: [
      {
        test: /\.ts$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
    ],
  },
  externals: [nodeExternals()],
  optimization: {
    minimize: process.env.NODE_ENV === 'production',
  },
};
