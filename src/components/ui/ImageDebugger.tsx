'use client';

import { useState } from 'react';
import { cleanImageUrl, getAlternativeImageUrls } from '@/utils/imageUtils';
import TokenImage from './TokenImage';

export default function ImageDebugger() {
  const [testUrl, setTestUrl] = useState('https://cdn.helius-rpc.com/cdn-cgi/image//https://ipfs.io/ipfs/bafkreicxm5z6te3677cy2dkpvsuvcu3nopawjbizkfswq5ten7sarobojy');
  const [results, setResults] = useState<any[]>([]);

  const testUrl1 = 'https://cdn.helius-rpc.com/cdn-cgi/image//https://ipfs.io/ipfs/bafkreicxm5z6te3677cy2dkpvsuvcu3nopawjbizkfswq5ten7sarobojy';
  const testUrl2 = 'https://ipfs.io/ipfs/bafkreicxm5z6te3677cy2dkpvsuvcu3nopawjbizkfswq5ten7sarobojy';

  const testUrls = () => {
    console.log('🧪 Iniciando prueba de URLs...');
    
    const originalUrl = testUrl;
    const cleanedUrl = cleanImageUrl(originalUrl);
    const alternativeUrls = getAlternativeImageUrls(originalUrl);
    
    console.log('📋 Resultados:');
    console.log('Original:', originalUrl);
    console.log('Cleaned:', cleanedUrl);
    console.log('Alternatives:', alternativeUrls);
    
    setResults([
      { type: 'Original', url: originalUrl },
      { type: 'Cleaned', url: cleanedUrl },
      ...alternativeUrls.map((url, index) => ({ type: `Alternative ${index + 1}`, url }))
    ]);
  };

  return (
    <div className="p-6 bg-white rounded-lg shadow-lg max-w-4xl mx-auto">
      <h2 className="text-2xl font-bold mb-4">🧪 Image URL Debugger</h2>
      
      <div className="mb-4">
        <label className="block text-sm font-medium mb-2">Test URL:</label>
        <input
          type="text"
          value={testUrl}
          onChange={(e) => setTestUrl(e.target.value)}
          className="w-full p-2 border border-gray-300 rounded-md"
          placeholder="Enter image URL to test"
        />
      </div>
      
      <div className="mb-4">
        <button
          onClick={testUrls}
          className="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600 mr-2"
        >
          Test URLs
        </button>
        <button
          onClick={() => setTestUrl(testUrl1)}
          className="bg-red-500 text-white px-4 py-2 rounded-md hover:bg-red-600 mr-2"
        >
          Test Broken URL
        </button>
        <button
          onClick={() => setTestUrl(testUrl2)}
          className="bg-green-500 text-white px-4 py-2 rounded-md hover:bg-green-600"
        >
          Test Direct IPFS
        </button>
      </div>
      
      {results.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800">🎯 Results</h3>
          <div className="space-y-2">
            {results.map((result, index) => (
              <div key={index} className="group relative bg-white border border-gray-200 rounded-lg p-3 hover:shadow-md transition-all duration-200">
                <div className="flex items-center gap-3">
                  {/* Imagen pequeña con hover escalable */}
                  <div className="relative flex-shrink-0">
                    <div className="w-10 h-10 bg-gray-100 rounded-lg border-2 border-gray-200 overflow-hidden group-hover:border-blue-300 transition-colors">
                      <TokenImage
                        src={result.url}
                        alt={`Test ${result.type}`}
                        className="w-full h-full object-cover group-hover:scale-[3] transition-transform duration-500 cursor-pointer"
                        tokenName="Test Token"
                        tokenSymbol="TEST"
                      />
                    </div>
                    {/* Indicador de zoom */}
                    <div className="absolute -top-1 -right-1 w-3 h-3 bg-blue-500 rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
                      <span className="text-white text-xs">🔍</span>
                    </div>
                  </div>
                  
                  {/* Información del resultado */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="inline-block px-2 py-1 bg-gradient-to-r from-blue-500 to-purple-600 text-white text-xs font-medium rounded-full">
                        {result.type}
                      </span>
                      <span className="text-xs text-green-600 font-medium">✓ Active</span>
                    </div>
                    <div className="text-xs text-gray-500 truncate group-hover:text-gray-700 transition-colors">
                      {result.url}
                    </div>
                  </div>
                  
                  {/* Botón de copia */}
                  <button 
                    onClick={() => navigator.clipboard.writeText(result.url)}
                    className="flex-shrink-0 w-8 h-8 bg-gray-100 hover:bg-gray-200 rounded-lg flex items-center justify-center transition-colors opacity-0 group-hover:opacity-100"
                    title="Copy URL"
                  >
                    <span className="text-xs">📋</span>
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      <div className="mt-6 p-4 bg-gray-100 rounded-md">
        <h4 className="font-medium mb-2">Quick Tests:</h4>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-sm font-medium">Problematic URL:</p>
            <TokenImage
              src={testUrl1}
              alt="Test Broken"
              className="w-20 h-20 rounded-md border mt-2"
              tokenName="duk"
              tokenSymbol="duk"
            />
          </div>
          <div>
            <p className="text-sm font-medium">Direct IPFS:</p>
            <TokenImage
              src={testUrl2}
              alt="Test Direct"
              className="w-20 h-20 rounded-md border mt-2"
              tokenName="duk"
              tokenSymbol="duk"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
