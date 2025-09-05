// @ts-nocheck
'use client';
import { useEffect, useRef } from 'react';
import { animate, scroll, spring } from 'motion';
import { motion, useScroll, useTransform } from 'framer-motion';

// Función para logs de imágenes
const logImageLoad = (imageName: string, success: boolean, error?: string) => {
  const timestamp = new Date().toISOString();
  if (success) {
    console.log(`✅ [${timestamp}] IMAGEN CARGADA: ${imageName}`);
  } else {
    console.error(`❌ [${timestamp}] ERROR AL CARGAR IMAGEN: ${imageName}`, error);
  }
};

export default function HorizontalScroll(): JSX.Element {
  const ulRef = useRef<HTMLUListElement | null>();
  const sectionRef = useRef<HTMLElement | null>();
  
  // Scroll progress específico para esta sección horizontal
  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ["start center", "end center"] // Desde que entra hasta que sale del centro
  });
  
  // Animaciones del robot - trigger específico para la card Precision
  // Se activa cuando la card Precision está visible y se scrollea horizontalmente
  const robotY = useTransform(scrollYProgress, [0.0, 0.15, 0.3], [-200, 0, 65]); // De 200px arriba a 200px abajo
  const robotScale = useTransform(scrollYProgress, [0.0, 0.15, 0.3], [0.8, 1, 1]);
  
  // Animaciones del logo - trigger específico para la card Precision
  // Se activa ligeramente antes que el robot para secuencia natural
  const logoX = useTransform(scrollYProgress, [0.0, 0.12, 0.25], [-200, 0, 0]);
  const logoY = useTransform(scrollYProgress, [0.0, 0.12, 0.25], [-100, 0, 0]);
  const logoOpacity = useTransform(scrollYProgress, [0.0, 0.12, 0.25], [0.5, 1, 1]);
  
// Animaciones para los logos en la sección PROFIT - Responsive basado en 1920x1080
const logo1X = useTransform(scrollYProgress, [0.0, 0.5], 
  [typeof window !== 'undefined' ? -575 * (window.innerWidth / 1920) : -575, 
   typeof window !== 'undefined' ? -675 * (window.innerWidth / 1920) : -675]); // Mover logo izquierdo más a la izquierda
const logo1Rotate = useTransform(scrollYProgress, [0.0, 0.5], [0, -15]); // Rotación más evidente izquierda
const logo2Y = useTransform(scrollYProgress, [0.0, 0.5], 
  [typeof window !== 'undefined' ? -700 * (window.innerHeight / 1080) : -700, 
   typeof window !== 'undefined' ? -800 * (window.innerHeight / 1080) : -800]); // Mover logo derecho más arriba
const logo2Rotate = useTransform(scrollYProgress, [0.0, 0.5], [15, 30]); // Rotación más evidente derecha (desde 15° base)

// Animaciones para la sección Speed - logo3Dsecond.png
// Se activa inmediatamente después de que Precision termina (0.3) - RANGO CORREGIDO
const speedLogoX = useTransform(scrollYProgress, [0.35, 0.50, 0.65], [800, 0, -800]); // Movimiento más agresivo: empieza derecha, va a izquierda
const speedLogoY = useTransform(scrollYProgress, [0.35, 0.50, 0.65], [200, 200, 200]); // 200px abajo del centro
const speedLogoRotate = useTransform(scrollYProgress, [0.35, 0.65], [0, -117]); // Rotación 35% más lenta (180 * 0.65 = 117 grados)

  useEffect(() => {
    const items = document.querySelectorAll('.section-scroll-horizontal li');
    const horizontalSection = document.querySelector('.section-scroll-horizontal');

    if (ulRef.current && horizontalSection) {
      const controls = animate(
        ulRef.current,
        {
          transform: ['none', `translateX(-${(items.length - 1) * 110}vw)`],
        },
        { easing: spring() }
      );
      
      scroll(controls, { target: horizontalSection });
      
      // Animaciones de headers
      const segmentLength = 1 / items.length;
      items.forEach((item, i) => {
        const header = item.querySelector('h2');
        scroll(animate([header], { x: [800, -800] }), {
          target: horizontalSection,
          offset: [
            [i * segmentLength, 1],
            [(i + 1) * segmentLength, 0],
          ],
        });
      });
    }
  }, []);

  return (
    <main>
        <article>
         
          <section ref={sectionRef} className='section-scroll-horizontal h-[400vh] relative'>
            <div className='sticky top-0 h-screen overflow-hidden'>
              <ul ref={ulRef} className='flex w-full h-full'>
              <li className='h-screen w-[110vw] min-w-[110vw] flex-shrink-0 bg-black flex flex-col justify-center overflow-hidden items-center relative'>
                {/* Logo 3D - solo animación de scroll */}
                <motion.div 
                  className='absolute left-[162px] top-1/2 transform -translate-y-1/2 translate-y-[50px]'
                  style={{ 
                    x: logoX, 
                    y: logoY, 
                    opacity: logoOpacity 
                  }}
                >
                  <img 
                    src='/logo3Dfour.png' 
                    alt='Logo 3D Second'
                    className='w-48 h-48 md:w-72 md:h-72 lg:w-96 lg:h-96 object-contain'
                    onLoad={() => logImageLoad('logo3Dfour.png', true)}
                    onError={(e) => logImageLoad('logo3Dfour.png', false, e.toString())}
                  />
                </motion.div>
                
                {/* Título mejorado - Responsive y mejor layout */}
                <h2 className='text-[14vw] sm:text-[16vw] md:text-[18vw] lg:text-[20vw] font-bold relative bottom-5 inline-block text-white whitespace-nowrap' style={{ fontFamily: 'Formula Condensed, sans-serif' }}>
                  Precision.
                </h2>
                
                {/* Robot - solo animación de scroll */}
                <motion.div 
                  className='absolute bottom-0 left-1/2 transform -translate-x-1/2'
                  style={{ 
                    y: robotY, 
                    scale: robotScale
                  }}
                >
                  <img 
                    src='/robot.png' 
                    alt='Robot'
                    className='w-[24.3rem] h-[24.3rem] md:w-[32.4rem] md:h-[32.4rem] lg:w-[40.5rem] lg:h-[40.5rem] object-contain'
                    onLoad={() => logImageLoad('robot.png', true)}
                    onError={(e) => logImageLoad('robot.png', false, e.toString())}
                  />
                </motion.div>
              </li>
              <li className='h-screen w-[110vw] min-w-[110vw] flex-shrink-0 bg-white flex flex-col justify-center overflow-hidden items-center relative'>
                {/* Logo 3D Second - animación de scroll hacia la izquierda */}
                <motion.div 
                  className='absolute left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2 z-10'
                  style={{ 
                    x: speedLogoX, 
                    y: speedLogoY, 
                    rotate: speedLogoRotate,
                    opacity: 1.0 
                  }}
                >
                  <img 
                    src='/logo3Dthree.png' 
                    alt='Logo 3D Second'
                    className='w-[476px] h-[476px] md:w-[764px] md:h-[764px] lg:w-[952px] lg:h-[952px] object-contain'
                    onLoad={() => logImageLoad('logo3Dthree.png', true)}
                    onError={(e) => logImageLoad('logo3Dthree.png', false, e.toString())}
                  />
                </motion.div>
                
                {/* Título mejorado - Responsive y mejor layout */}
                <h2 className='text-[14vw] sm:text-[16vw] md:text-[18vw] lg:text-[20vw] font-bold relative bottom-5 inline-block text-black whitespace-nowrap' style={{ fontFamily: 'Formula Condensed, sans-serif' }}>
                  Speed.
                </h2>
              </li>
              <li className='h-screen w-[110vw] min-w-[110vw] flex-shrink-0 flex flex-col justify-center overflow-hidden  items-center relative' style={{ backgroundColor: '#BEF135' }}>
                <h2 className='text-[14vw] sm:text-[16vw] md:text-[18vw] lg:text-[20vw] font-bold relative bottom-5 inline-block text-black whitespace-nowrap' style={{ fontFamily: 'Formula Condensed, sans-serif' }}>
                  PROFIT.
                </h2>
                
                {/* Container para las imágenes - phone.png en el centro y logo3D.png a los lados */}
                <div className='absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 translate-y-8'>
                  {/* Logo 3D izquierdo - Responsive basado en 1920x1080 */}
                  <motion.img 
                    src='/logo3D.png' 
                    alt='Logo 3D Left' 
                    className='absolute object-contain' 
                    style={{
                      x: logo1X,
                      rotate: logo1Rotate,
                      scale: typeof window !== 'undefined' ? 1.35 * Math.min(window.innerWidth / 1920, window.innerHeight / 1080) : 1.35,
                      width: typeof window !== 'undefined' ? `${320 * (window.innerWidth / 1920)}px` : '320px',
                      height: typeof window !== 'undefined' ? `${320 * (window.innerWidth / 1920)}px` : '320px',
                      transformOrigin: 'center'
                    }}
                    onLoad={() => logImageLoad('logo3D.png (izquierdo)', true)}
                    onError={(e) => logImageLoad('logo3D.png (izquierdo)', false, e.toString())}
                  />
                  
                  {/* Teléfono en el centro - Responsive basado en 1920x1080 - Oculto en móviles */}
                  <img 
                    src='/phone.png' 
                    alt='Phone Image' 
                    className='object-contain hidden md:block' 
                    style={{
                      transform: typeof window !== 'undefined' ? 
                        `translate(${-150 * (window.innerWidth / 1920)}px, ${(-50 + 30 + 15) * (window.innerHeight / 1080)}px) scale(${1.5 * Math.min(window.innerWidth / 1920, window.innerHeight / 1080)})` : 
                        'translate(-150px, -5px) scale(1.5)',
                      width: typeof window !== 'undefined' ? `${384 * (window.innerWidth / 1920)}px` : '384px',
                      height: typeof window !== 'undefined' ? `${384 * (window.innerWidth / 1920)}px` : '384px',
                      transformOrigin: 'center'
                    }}
                    onLoad={() => logImageLoad('phone.png', true)}
                    onError={(e) => logImageLoad('phone.png', false, e.toString())}
                  />
                  
                  {/* Logo 3D derecho - Responsive basado en 1920x1080 */}
                  <motion.img 
                    src='/logo3D.png' 
                    alt='Logo 3D Right' 
                    className='absolute object-contain' 
                    style={{
                      x: typeof window !== 'undefined' ? 250 * (window.innerWidth / 1920) : 250,
                      y: logo2Y,
                      rotate: logo2Rotate,
                      scale: typeof window !== 'undefined' ? 1.35 * Math.min(window.innerWidth / 1920, window.innerHeight / 1080) : 1.35,
                      width: typeof window !== 'undefined' ? `${320 * (window.innerWidth / 1920)}px` : '320px',
                      height: typeof window !== 'undefined' ? `${320 * (window.innerWidth / 1920)}px` : '320px',
                      transformOrigin: 'center'
                    }}
                    onLoad={() => logImageLoad('logo3D.png (derecho)', true)}
                    onError={(e) => logImageLoad('logo3D.png (derecho)', false, e.toString())}
                  />
                </div>
              </li>
            </ul>
            </div>
          </section>
          
        </article>
        <div className='progress fixed left-0 right-0  h-2 rounded-full bg-red-600 bottom-[50px] scale-x-0'></div>
      </main>
  );
}
