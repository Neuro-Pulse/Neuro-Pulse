'use client';

import { cleanImageUrl, logTokenImageLoad } from '@/utils/imageUtils';

interface TokenImageProps {
  src: string;
  alt: string;
  className?: string;
  tokenName?: string;
  tokenSymbol?: string;
  onLoad?: () => void;
  onError?: (error: any) => void;
}

export default function TokenImage({
  src,
  alt,
  className = '',
  tokenName = 'Unknown',
  tokenSymbol = '???',
  onLoad,
  onError
}: TokenImageProps) {
  // Limpiar URL automáticamente
  const cleanedSrc = cleanImageUrl(src);
  
  // Log automático cuando se intenta cargar la imagen
  if (src !== cleanedSrc) {
    console.log(`🔧 [TokenImage] URL corregida para ${tokenName} (${tokenSymbol}):`);
    console.log(`❌ Original: ${src}`);
    console.log(`✅ Corregida: ${cleanedSrc}`);
  }

  const handleLoad = () => {
    logTokenImageLoad(tokenName, tokenSymbol, src, true, cleanedSrc);
    onLoad?.();
  };

  const handleError = (event: any) => {
    logTokenImageLoad(tokenName, tokenSymbol, src, false, cleanedSrc, event);
    onError?.(event);
  };

  return (
    <img
      src={cleanedSrc}
      alt={alt}
      className={className}
      onLoad={handleLoad}
      onError={handleError}
    />
  );
}
