"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useState, useEffect } from "react";

interface PreLoaderProps {
  onComplete: () => void;
}

export const PreLoader = ({ onComplete }: PreLoaderProps) => {
  const [phase, setPhase] = useState<'growing' | 'expanding' | 'revealing' | 'complete'>('growing');

  useEffect(() => {
    // Fase 1: Crecimiento vertical (2.7s - 0.5s más para inicio más lento)
    const growingTimer = setTimeout(() => {
      setPhase('expanding');
      
      // Fase 2: Expansión horizontal (0.4s)
      const expandingTimer = setTimeout(() => {
        setPhase('revealing');
        
        // Fase 3: Revelación del contenido (empieza al 70% de la expansión)
        const revealTimer = setTimeout(() => {
          setPhase('complete');
          onComplete();
        }, 300); // Tiempo adicional para completar la revelación
        
        return () => clearTimeout(revealTimer);
      }, 400); // Duración de la expansión
      
      return () => clearTimeout(expandingTimer);
    }, 2700); // Duración del crecimiento vertical aumentada en 0.5s

    return () => clearTimeout(growingTimer);
  }, [onComplete]);

  return (
    <AnimatePresence mode="wait">
      {phase !== 'complete' && (
        <motion.div
          className="fixed inset-0 z-[99999] flex items-center justify-center overflow-hidden"
          style={{ backgroundColor: "rgb(0, 0, 0)" }}
          initial={{ opacity: 1 }}
          animate={{ opacity: phase === 'revealing' ? 0 : 1 }}
          exit={{ opacity: 0 }}
          transition={{ 
            duration: phase === 'revealing' ? 0.6 : 0.3,
            ease: [0.7, 0, 0.3, 1] // Mismo easing rápido al inicio, lento al final
          }}
        >
          {/* Palito central que crece verticalmente y luego se expande */}
          <motion.div
            className="bg-white"
            initial={{ 
              width: 2, 
              height: 0,
            }}
            animate={{
              width: phase === 'expanding' || phase === 'revealing' ? "100vw" : 2,
              height: phase === 'growing' ? [
                0,        // 0% - Inicio
                "12vh",   // Salto 1
                "8vh",    // Bajada 1
                "8vh",    // Pausa
                "25vh",   // Salto 2
                "22vh",   // Bajada 2
                "22vh",   // Pausa
                "42vh",   // Salto 3
                "38vh",   // Bajada 3
                "38vh",   // Pausa
                "58vh",   // Salto 4
                "54vh",   // Bajada 4
                "54vh",   // Pausa
                "75vh",   // Salto 5
                "72vh",   // Bajada 5
                "72vh",   // Pausa
                "90vh",   // Salto 6
                "87vh",   // Bajada 6
                "87vh",   // Pausa corta
                "100vh"   // Salto final al 100%
              ] : "100vh",
              transition: {
                width: {
                  duration: phase === 'expanding' ? 0.4 : 0,
                  ease: [0.7, 0, 0.3, 1], // Rápido al inicio, lento al final (easeOut fuerte)
                },
                height: {
                  duration: phase === 'growing' ? 2.7 : 0, // Aumentado a 2.7s
                  ease: [0, 0, 1, 1], // Saltos completamente instantáneos, sin transición
                  times: phase === 'growing' ? [
                    0,     // Inicio
                    0.08,  // Salto 1 - mucho más lento (antes 0.04)
                    0.09,  // Bajada 1 instantánea
                    0.15,  // Pausa más larga
                    0.20,  // Salto 2 - más lento (antes 0.14)
                    0.21,  // Bajada 2 instantánea
                    0.28,  // Pausa
                    0.33,  // Salto 3 - más lento (antes 0.27)
                    0.34,  // Bajada 3 instantánea
                    0.42,  // Pausa media
                    0.48,  // Salto 4 - más gradual
                    0.50,  // Bajada 4
                    0.60,  // Pausa más larga
                    0.66,  // Salto 5
                    0.68,  // Bajada 5
                    0.80,  // Pausa larga
                    0.88,  // Salto 6
                    0.90,  // Bajada 6
                    0.96,  // Pausa final
                    1      // Salto final al 100vh
                  ] : undefined,
                }
              }
            }}
            style={{
              originY: 0.5,
              originX: 0.5,
            }}
          />
        </motion.div>
      )}
    </AnimatePresence>
  );
};
