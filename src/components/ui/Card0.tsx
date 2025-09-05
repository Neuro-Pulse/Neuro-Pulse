'use client';

import { motion, MotionValue, useAnimation } from 'framer-motion';
import { useTransform } from 'framer-motion';
import { useState, useEffect } from 'react';

interface Card0Props {
  scrollYProgress: MotionValue<number>;
}

export default function Card0({ scrollYProgress }: Card0Props) {
  // Card0 con STICKY EFFECT MEJORADO - 210vh con trigger ajustado:
  // 0-10%: slide-in extremadamente lento y suave, 10-15%: SE PEGA FUERTEMENTE A LA PANTALLA, 15-17%: transición suave a verde
  const card0SlideX = useTransform(scrollYProgress, [0, 0.10, 0.25], ['25%', '-25%', '-25%']);
  const card0SlideY = useTransform(scrollYProgress, [0, 0.10, 0.25], ['20%', '0%', '0%']);

  const backgroundControls = useAnimation();
  const contentControls = useAnimation();
  const [shouldAnimate, setShouldAnimate] = useState(false);

  useEffect(() => {
    const unsubscribe = scrollYProgress.on("change", (value) => {
      if (value >= 0.157 && !shouldAnimate) {
        setShouldAnimate(true);
        // Fondo se vuelve verde PERMANENTEMENTE
        backgroundControls.start({ backgroundColor: '#C3F73A' });
        // Contenido desaparece
        contentControls.start({ opacity: 0 });
      } else if (value < 0.157 && shouldAnimate) {
        setShouldAnimate(false);
        // Fondo vuelve a gris solo si vamos hacia atrás
        backgroundControls.start({ backgroundColor: '#EBEFEF' });
        // Contenido reaparece
        contentControls.start({ opacity: 1 });
      }
    });
    return unsubscribe;
  }, [scrollYProgress, shouldAnimate, backgroundControls, contentControls]);

  return (
    <div className='h-[210vh] w-full sticky top-0 overflow-hidden' style={{ clipPath: 'inset(0 0 0 0)' }}>
      <motion.section 
        className='h-full w-[150vw] flex flex-col lg:flex-row rounded-tl-[25.3rem] overflow-hidden'
        style={{ x: card0SlideX, y: card0SlideY }}
        transition={{ 
          type: "spring", 
          stiffness: 5, 
          damping: 60,
          ease: "easeInOut"
        }}
      >
      {/* Fondo completo */}
      <motion.div 
        className='w-full h-full rounded-tl-[6.3rem] relative'
        animate={backgroundControls}
        initial={{ backgroundColor: '#EBEFEF' }}
      >
        
        {/* DESKTOP (1920x1080) - Layout horizontal perfecto */}
        <div className='hidden lg:block'>
          {/* Texto posicionado absolutamente - 450px arriba con separación y MÁS ANCHO - 100px a la derecha */}
          <motion.div 
            className='absolute top-0 left-0 w-1/2 h-full flex items-center justify-end pr-12' 
            style={{ transform: 'translateY(-450px) translateX(100px)' }} animate={contentControls}
          >
            <div className='flex flex-col max-w-lg'>
              {/* Título con HelveticaNowText-Black-Demo */}
              <h2 className='text-black text-lg font-black mb-4'
                  style={{ fontFamily: 'HelveticaNowText-Black-Demo' }}>
                ▶ INTRODUCTION
              </h2>
              
              {/* Descripción con Helvetica Now Text - MÁS ANCHO para menos líneas */}
              <p className='text-black text-2xl font-medium max-w-lg'
                 style={{ fontFamily: 'Helvetica Now Text', lineHeight: '1.68' }}>
                We scan thousands of wallets across blockchains to detect smart money movements. When top traders buy or sell, you get instant alerts. Simple, fast, powerful.
              </p>
            </div>
          </motion.div>

          {/* Imagen posicionada absolutamente - 450px arriba con separación - 25% MEN0S - 250px a la derecha */}
          <motion.div 
            className='absolute top-0 right-0 w-1/2 h-full flex items-center justify-start pl-12' 
            style={{ transform: 'translateY(-450px) translateX(250px)' }} animate={contentControls}
          >
            <div className='w-full max-w-xl h-[90vh] rounded-2xl overflow-hidden'>
              <img 
                src='/logo3Dfour.png' 
                alt='Introduction Image'
                className='w-full h-full object-cover'
              />
            </div>
          </motion.div>
        </div>
        
        {/* MÓVIL/TABLET - Layout vertical simple */}
        <div className='block lg:hidden'>
          <div className='flex flex-col items-center justify-center h-full px-6 py-8'>
            {/* Título móvil */}
            <h2 className='text-black text-base font-black mb-4 text-center'
                style={{ fontFamily: 'HelveticaNowText-Black-Demo' }}>
              ▶ INTRODUCTION
            </h2>
            
            {/* Imagen móvil */}
            <div className='w-full max-w-sm h-[40vh] rounded-2xl overflow-hidden mb-6'>
              <img 
                src='/logo3Dfour.png' 
                alt='Introduction Image'
                className='w-full h-full object-cover'
              />
            </div>
            
            {/* Texto móvil */}
            <p className='text-black text-base font-medium text-center max-w-sm'
               style={{ fontFamily: 'Helvetica Now Text', lineHeight: '1.5' }}>
              We scan thousands of wallets across blockchains to detect smart money movements. When top traders buy or sell, you get instant alerts. Simple, fast, powerful.
            </p>
          </div>
        </div>
        
      </motion.div>
      </motion.section>
    </div>
  );
}
