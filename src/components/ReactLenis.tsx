'use client';

import { useEffect, useRef, ReactNode } from 'react';
import Lenis from 'lenis';

interface ReactLenisProps {
  children: ReactNode;
  options?: {
    duration?: number;
    easing?: (t: number) => number;
    smoothWheel?: boolean;
    smoothTouch?: boolean;
    wheelMultiplier?: number;
    touchMultiplier?: number;
    syncTouch?: boolean;
  };
}

export default function ReactLenis({ children, options = {} }: ReactLenisProps) {
  const lenisRef = useRef<Lenis | null>(null);

  useEffect(() => {
    // Solo ejecutar en el cliente
    if (typeof window === 'undefined') return;

    // Configuración por defecto sutil
    const defaultOptions = {
      duration: 1.2,
      easing: (t: number) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      smoothWheel: true,
      smoothTouch: false, // Deshabilitado para mejor performance en móvil
      wheelMultiplier: 0.8,
      touchMultiplier: 1.5,
      syncTouch: true,
    };

    // Combinar opciones
    const lenisOptions = { ...defaultOptions, ...options };

    // Crear instancia de Lenis
    const lenis = new Lenis({
      duration: lenisOptions.duration,
      easing: lenisOptions.easing,
      wheelMultiplier: lenisOptions.wheelMultiplier,
      touchMultiplier: lenisOptions.touchMultiplier,
      infinite: false,
    });

    lenisRef.current = lenis;

    // Función de animación
    function raf(time: number) {
      lenis.raf(time);
      requestAnimationFrame(raf);
    }

    requestAnimationFrame(raf);

    // Cleanup
    return () => {
      if (lenisRef.current) {
        lenisRef.current.destroy();
        lenisRef.current = null;
      }
    };
  }, [options]);

  return <>{children}</>;
}
