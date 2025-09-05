'use client';

import { motion, useScroll, useTransform } from 'framer-motion';
import { useRef } from 'react';

interface ParallaxCardProps {
  cardNumber: string;
  totalCards: number;
  backgroundColor: string;
  imageSrc: string;
  clipPath?: any;
  titleAnimationRange: [number, number];
  containerRef: React.RefObject<HTMLDivElement>;
}

export default function ParallaxCard({
  cardNumber,
  totalCards,
  backgroundColor,
  imageSrc,
  clipPath,
  titleAnimationRange,
  containerRef
}: ParallaxCardProps) {
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end start"]
  });

  // Animación del título específica para esta card
  const titleY = useTransform(scrollYProgress, titleAnimationRange, ["100%", "0%"]);

  return (
    <motion.section 
      className='h-screen w-full flex flex-col lg:flex-row sticky top-0'
      style={{ clipPath }}
    >
      {/* Left Section - Text Block */}
      <div className={`flex-1 p-4 sm:p-6 md:p-8 lg:p-12 xl:p-16 flex flex-col justify-start pt-[6rem] sm:pt-[7rem] md:pt-[8rem] lg:pt-[9rem] xl:pt-[10rem] xl:pt-[11rem] relative min-h-[50vh] lg:min-h-screen`} style={{ backgroundColor }}>
        <div className='flex flex-col'>
          {/* Contenedores con overflow hidden individual para cada línea */}
          <div className='text-black text-left leading-[0.65] sm:leading-[0.7] md:leading-[0.75] lg:leading-[0.65] mb-2 sm:mb-3 md:mb-4' style={{ fontFamily: 'Formula Condensed', fontWeight: 900 }}>
            {/* Primera línea: {cardNumber} LIQUID */}
            <div className='overflow-hidden mb-1'>
              <motion.div 
                className='flex flex-col sm:flex-row sm:items-center sm:gap-3 md:gap-4 lg:gap-5 xl:gap-6'
                style={{ y: titleY }}
              >
                <span className='text-white text-3xl sm:text-4xl md:text-6xl lg:text-7xl xl:text-8xl 2xl:text-9xl 3xl:text-10xl' style={{ fontFamily: 'Formula Condensed', fontWeight: 900 }}>{cardNumber}</span>
                <span className='text-3xl sm:text-4xl md:text-6xl lg:text-7xl xl:text-8xl 2xl:text-9xl 3xl:text-10xl' style={{ fontFamily: 'Formula Condensed', fontWeight: 900 }}>LIQUID</span>
              </motion.div>
            </div>
            
            {/* Segunda línea: GAMING */}
            <div className='overflow-hidden mb-1'>
              <motion.div 
                className='text-3xl sm:text-4xl md:text-6xl lg:text-7xl xl:text-8xl 2xl:text-9xl 3xl:text-10xl'
                style={{ 
                  fontFamily: 'Formula Condensed', 
                  fontWeight: 900,
                  y: titleY
                }}
              >
                GAMING
              </motion.div>
            </div>
            
            {/* Tercera línea: ECONOMY */}
            <div className='overflow-hidden'>
              <motion.div 
                className='text-3xl sm:text-4xl md:text-6xl lg:text-7xl xl:text-8xl 2xl:text-9xl 3xl:text-10xl'
                style={{ 
                  fontFamily: 'Formula Condensed', 
                  fontWeight: 900,
                  y: titleY
                }}
              >
                ECONOMY
              </motion.div>
            </div>
          </div>
          
          <p className='text-black text-sm sm:text-base md:text-lg lg:text-xl max-w-full sm:max-w-lg md:max-w-xl lg:max-w-2xl mt-4 sm:mt-6 md:mt-8 leading-relaxed' style={{ fontFamily: 'Helvetica Now Text', fontWeight: 400 }}>
            Recapture the 95%. Non-monetizing players earn $PLAI to spend in-game, 
            bypassing predatory play-to-earn models. Developers integrating $PLAI as 
            a native payment option see even better conversions.
          </p>
        </div>
        
        <div className='absolute bottom-4 sm:bottom-6 md:bottom-8 lg:bottom-12 xl:bottom-16 left-4 sm:left-6 md:left-8 lg:left-12 xl:left-16 text-black text-xs font-medium'>
          {cardNumber === '00' ? '0' : cardNumber.replace('0', '')}/{totalCards}
        </div>
      </div>
      
      {/* Right Section - Image Block */}
      <div className={`flex-1 p-4 sm:p-6 md:p-8 lg:p-12 xl:p-16 flex items-center justify-center min-h-[50vh] lg:min-h-screen`} style={{ backgroundColor }}>
        <div className='w-full max-w-md sm:max-w-lg md:max-w-xl lg:max-w-2xl xl:max-w-5xl h-[31vh] sm:h-[38vh] md:h-[47vh] lg:h-[54vh] xl:h-[62vh] rounded-2xl sm:rounded-3xl overflow-hidden'>
          <img 
            src={imageSrc} 
            alt={`Card ${cardNumber} Image`}
            className='w-full h-full object-cover'
          />
        </div>
      </div>
    </motion.section>
  );
}
