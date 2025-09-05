// Utilidades para manejar URLs de imágenes de tokens

/**
 * Limpia y corrige URLs de imágenes problemáticas
 */
export const cleanImageUrl = (url: string): string => {
  if (!url) return '';
  
  // Buscar hash de IPFS directamente al final de la URL
  const ipfsHashMatch = url.match(/bafkrei[a-z0-9]+/);
  if (ipfsHashMatch) {
    return `https://ipfs.io/ipfs/${ipfsHashMatch[0]}`;
  }
  
  // Si no hay hash de IPFS, devolver la URL original
  return url;
};

/**
 * Obtiene URLs alternativas para una imagen de token
 */
export const getAlternativeImageUrls = (originalUrl: string): string[] => {
  const alternatives: string[] = [];
  
  if (!originalUrl) return alternatives;
  
  // Agregar la URL original limpia
  alternatives.push(cleanImageUrl(originalUrl));
  
  // Si es una URL de Helius CDN, agregar la URL directa de IPFS
  if (originalUrl.includes('cdn.helius-rpc.com') && originalUrl.includes('ipfs.io')) {
    const ipfsMatch = originalUrl.match(/https:\/\/ipfs\.io\/ipfs\/[^\/]+/);
    if (ipfsMatch) {
      alternatives.push(ipfsMatch[0]);
    }
  }
  
  // Agregar gateway alternativo de IPFS
  if (originalUrl.includes('ipfs.io/ipfs/')) {
    const hash = originalUrl.split('/ipfs/')[1];
    if (hash) {
      alternatives.push(`https://gateway.pinata.cloud/ipfs/${hash}`);
      alternatives.push(`https://cloudflare-ipfs.com/ipfs/${hash}`);
    }
  }
  
  return [...new Set(alternatives)]; // Eliminar duplicados
};

/**
 * Logs específicos para imágenes de tokens
 */
export const logTokenImageLoad = (
  tokenName: string,
  tokenSymbol: string,
  originalUrl: string,
  success: boolean,
  finalUrl?: string,
  error?: any
) => {
  const timestamp = new Date().toISOString();
  const tokenInfo = `${tokenName} (${tokenSymbol})`;
  
  if (success) {
    console.log(`✅ [${timestamp}] TOKEN IMAGE LOADED: ${tokenInfo}`);
    console.log(`🔗 Original URL: ${originalUrl}`);
    if (finalUrl && finalUrl !== originalUrl) {
      console.log(`🔄 Final URL: ${finalUrl}`);
    }
  } else {
    console.error(`❌ [${timestamp}] TOKEN IMAGE FAILED: ${tokenInfo}`);
    console.error(`🔗 Failed URL: ${originalUrl}`);
    if (error) {
      console.error(`💥 Error details:`, error);
    }
  }
};

