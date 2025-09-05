"use client";

import { motion, useScroll, useTransform, useInView } from "framer-motion";
import { useRef } from "react";
import Card0 from "./Card0";

export default function ParallaxLayers() {
  const containerRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end start"],
  });

  // NUEVA SECUENCIA CON CARD 0 (200vh) - TIMELINE OPTIMIZADO PARA 800vh:
  // Altura total ajustada a 800vh = 200vh (Card0) + 600vh (resto)
  // Timeline optimizado para utilizar todo el scroll (termina al 0.99-1.0):
  // Carta 0: 200vh total - slide-in desde izquierda (0-22% del scroll)
  // Carta 1: visible desde 25%
  // Título 1: scrollea del 35% al 45%
  // Carta 2: se revela del 50% al 60%
  // Título 2: scrollea del 65% al 75%
  // Carta 3: se revela del 80% al 90% + título (92-98%)
  // Carta 4: se revela del 95% al 98% y permanece hasta 100%

  // Card0 ahora está en su propio componente

  // Fase 1: Efecto del título de la carta 1 (35-45% del scroll)
  const titleY = useTransform(scrollYProgress, [0.3, 0.4], ["0%", "-100%"]);

  // Fase 2: Reveal de la carta 2 (50-60% del scroll)
  const card2Clip = useTransform(
    scrollYProgress,
    [0.4, 0.5],
    [
      "polygon(0 100%, 100% 100%, 100% 100%, 0 100%)",
      "polygon(0 0%, 100% 0%, 100% 100%, 0 100%)",
    ]
  );

  // Fase 3: Efecto del título de la carta 2 (65-75% del scroll)
  const title2Y = useTransform(scrollYProgress, [0.5, 0.6], ["0%", "-100%"]);

  // Fase 4: Reveal de la carta 3 (80-90% del scroll)
  const card3Clip = useTransform(
    scrollYProgress,
    [0.6, 0.7],
    [
      "polygon(0 100%, 100% 100%, 100% 100%, 0 100%)",
      "polygon(0 0%, 100% 0%, 100% 100%, 0 100%)",
    ]
  );
  const title3Y = useTransform(scrollYProgress, [0.7, 0.8], ["0%", "-100%"]);

  // Fase 5: Reveal de la carta 4 blanca (95-98% del scroll) - Timing perfecto
  const card4Clip = useTransform(
    scrollYProgress,
    [0.79, 0.89],
    [
      "polygon(0 100%, 100% 100%, 100% 100%, 0 100%)",
      "polygon(0 0%, 100% 0%, 100% 100%, 0 100%)",
    ]
  );

  // Ref para la sección HOW IT WORKS
  const howItWorksRef = useRef<HTMLElement>(null);
  const isInView = useInView(howItWorksRef, { once: true, margin: "-20%" });

  // Scroll progress específico para la sección HOW IT WORKS
  const { scrollYProgress: howItWorksProgress } = useScroll({
    target: howItWorksRef,
    offset: ["start end", "end start"],
  });

  // Animación de escala para el contenedor orbital
  const orbitalScale = useTransform(howItWorksProgress, [0.3, 0.6], [0.6, 1.0]);

  return (
    <main className="bg-white">
      {/* Espacio buffer responsivo para que el sticky del marquee complete sus 70vh */}
      <div className="h-[50vh] sm:h-[60vh] md:h-[65vh] lg:h-[70vh]"></div>

      {/* Contenedor con altura para que las secciones sticky funcionen */}
      <article className="h-[800vh]" ref={containerRef}>
        {/* 200vh (Card0) + 600vh (resto) = 800vh total */}

        {/* Card 0 - 150vh de altura con slide-in animation */}
        {/* <div className='bg-white'> */}
        <Card0 scrollYProgress={scrollYProgress} />
        {/* </div> */}

        {/* Card 1 - Altura normal 100vh */}
        <section className="h-screen w-full flex flex-col lg:flex-row sticky top-0">
          {/* Left Section - Text Block */}
          <div className="flex-1 bg-[#C3F73A] p-4 sm:p-6 md:p-8 lg:p-12 xl:p-16 flex flex-col justify-start pt-[6rem] sm:pt-[7rem] md:pt-[8rem] lg:pt-[9rem] xl:pt-[10rem] xl:pt-[11rem] relative min-h-[50vh] lg:min-h-screen">
            <div className="flex flex-col">
              {/* Contenedores con overflow hidden individual para cada línea */}
              <div
                className="text-black text-left leading-[0.65] sm:leading-[0.7] md:leading-[0.75] lg:leading-[0.65] mb-2 sm:mb-3 md:mb-4"
                style={{ fontFamily: "Formula Condensed", fontWeight: 900 }}
              >
                {/* Primera línea: 01 LIQUID */}
                <div className="overflow-hidden mb-1">
                  <motion.div
                    className="flex flex-col sm:flex-row sm:items-center sm:gap-3 md:gap-4 lg:gap-5 xl:gap-6"
                    style={{ y: titleY }}
                  >
                    <span
                      className="text-white text-3xl sm:text-4xl md:text-6xl lg:text-7xl xl:text-8xl 2xl:text-9xl 3xl:text-10xl"
                      style={{
                        fontFamily: "Formula Condensed",
                        fontWeight: 900,
                      }}
                    >
                      01
                    </span>
                    <span
                      className="text-3xl sm:text-4xl md:text-6xl lg:text-7xl xl:text-8xl 2xl:text-9xl 3xl:text-10xl"
                      style={{
                        fontFamily: "Formula Condensed",
                        fontWeight: 900,
                      }}
                    >
                      PLUG
                    </span>
                  </motion.div>
                </div>

                {/* Segunda línea: GAMING */}
                <div className="overflow-hidden mb-1">
                  <motion.div
                    className="text-3xl sm:text-4xl md:text-6xl lg:text-7xl xl:text-8xl 2xl:text-9xl 3xl:text-10xl"
                    style={{
                      fontFamily: "Formula Condensed",
                      fontWeight: 900,
                      y: titleY,
                    }}
                  >
                    INTO
                  </motion.div>
                </div>

                {/* Tercera línea: ECONOMY */}
                <div className="overflow-hidden">
                  <motion.div
                    className="text-3xl sm:text-4xl md:text-6xl lg:text-7xl xl:text-8xl 2xl:text-9xl 3xl:text-10xl"
                    style={{
                      fontFamily: "Formula Condensed",
                      fontWeight: 900,
                      y: titleY,
                    }}
                  >
                    Neuro Pulse
                  </motion.div>
                </div>
              </div>

              <p
                className="text-black text-sm sm:text-base md:text-lg lg:text-xl max-w-full sm:max-w-lg md:max-w-xl lg:max-w-2xl mt-4 sm:mt-6 md:mt-8 leading-relaxed"
                style={{ fontFamily: "Helvetica Now Text", fontWeight: 400 }}
              >
                Most traders miss the signals that matter. $TECH connects you to the wallets that move Solana.
Install the Chrome extension or start via Telegram — no friction, no fluff.
Once linked, your feed lights up with real-time alpha from the smartest money on-chain.
              </p>
            </div>

            <div className="absolute bottom-4 sm:bottom-6 md:bottom-8 lg:bottom-12 xl:bottom-16 left-4 sm:left-6 md:left-8 lg:left-12 xl:left-16 text-black text-xs font-medium">
              1/4
            </div>
          </div>

          {/* Right Section - Image Block */}
          <div className="flex-1 p-4 sm:p-6 md:p-8 lg:p-12 xl:p-16 flex items-center justify-center bg-[#C3F73A] min-h-[50vh] lg:min-h-screen">
            <div className="w-full max-w-md sm:max-w-lg md:max-w-xl lg:max-w-2xl xl:max-w-5xl h-[31vh] sm:h-[38vh] md:h-[47vh] lg:h-[54vh] xl:h-[62vh] rounded-2xl sm:rounded-3xl overflow-hidden">
              <img
                src="/image1.png"
                alt="Card 1 Image"
                className="w-full h-full object-cover"
              />
            </div>
          </div>
        </section>

        <motion.section
          className="h-screen w-full flex flex-col lg:flex-row sticky top-0"
          style={{ clipPath: card2Clip }}
        >
          {/* Left Section - Text Block */}
          <div className="flex-1 bg-[#F36CD0] p-4 sm:p-6 md:p-8 lg:p-12 xl:p-16 flex flex-col justify-start pt-[7rem] sm:pt-[8rem] md:pt-[9rem] lg:pt-[10rem] xl:pt-[11rem] relative min-h-[50vh] lg:min-h-screen">
            <div className="flex flex-col">
              {/* Contenedores con overflow hidden individual para cada línea */}
              <div
                className="text-black text-left leading-[0.65] sm:leading-[0.7] md:leading-[0.75] lg:leading-[0.65] mb-2 sm:mb-3 md:mb-4"
                style={{ fontFamily: "Formula Condensed", fontWeight: 900 }}
              >
                {/* Primera línea: 02 LIQUID */}
                <div className="overflow-hidden mb-1">
                  <motion.div
                    className="flex flex-col sm:flex-row sm:items-center sm:gap-3 md:gap-4 lg:gap-5 xl:gap-6"
                    style={{ y: title2Y }}
                  >
                    <span
                      className="text-white text-3xl sm:text-4xl md:text-6xl lg:text-7xl xl:text-8xl 2xl:text-9xl 3xl:text-10xl"
                      style={{
                        fontFamily: "Formula Condensed",
                        fontWeight: 900,
                      }}
                    >
                      02
                    </span>
                    <span
                      className="text-3xl sm:text-4xl md:text-6xl lg:text-7xl xl:text-8xl 2xl:text-9xl 3xl:text-10xl"
                      style={{
                        fontFamily: "Formula Condensed",
                        fontWeight: 900,
                      }}
                    >
                      FOLLOW
                    </span>
                  </motion.div>
                </div>

                {/* Segunda línea: GAMING */}
                <div className="overflow-hidden mb-1">
                  <motion.div
                    className="text-3xl sm:text-4xl md:text-6xl lg:text-7xl xl:text-8xl 2xl:text-9xl 3xl:text-10xl"
                    style={{
                      fontFamily: "Formula Condensed",
                      fontWeight: 900,
                      y: title2Y,
                    }}
                  >
                    THE
                  </motion.div>
                </div>

                {/* Tercera línea: ECONOMY */}
                <div className="overflow-hidden">
                  <motion.div
                    className="text-3xl sm:text-4xl md:text-6xl lg:text-7xl xl:text-8xl 2xl:text-9xl 3xl:text-10xl"
                    style={{
                      fontFamily: "Formula Condensed",
                      fontWeight: 900,
                      y: title2Y,
                    }}
                  >
                    FLOW
                  </motion.div>
                </div>
              </div>

              <p
                className="text-black text-sm sm:text-base md:text-lg lg:text-xl max-w-full sm:max-w-lg md:max-w-xl lg:max-w-2xl mt-4 sm:mt-6 md:mt-8 leading-relaxed"
                style={{ fontFamily: "Helvetica Now Text", fontWeight: 400 }}
              >
                Stop chasing noise. Our AI filters wallet activity to surface only high-conviction moves.
Whether it's a new meme coin or a quiet accumulation, $TECH sees it first.
Add your wallets, sit back, and let the signal stream do the heavy lifting.
              </p>
            </div>

            <div className="absolute bottom-4 sm:bottom-6 md:bottom-8 lg:bottom-12 xl:bottom-16 left-4 sm:left-6 md:left-8 lg:left-12 xl:left-16 text-black text-xs font-medium">
              2/4
            </div>
          </div>

          {/* Right Section - Image Block */}
          <div className="flex-1 p-4 sm:p-6 md:p-8 lg:p-12 xl:p-16 flex items-center justify-center bg-[#F36CD0] min-h-[50vh] lg:min-h-screen">
            <div className="w-full max-w-md sm:max-w-lg md:max-w-xl lg:max-w-2xl xl:max-w-5xl h-[31vh] sm:h-[38vh] md:h-[47vh] lg:h-[54vh] xl:h-[62vh] rounded-2xl sm:rounded-3xl overflow-hidden">
              <img
                src="/image2.png"
                alt="Card 2 Image"
                className="w-full h-full object-cover"
              />
            </div>
          </div>
        </motion.section>

        <motion.section
          className="h-screen w-full flex flex-col lg:flex-row sticky top-0"
          style={{ clipPath: card3Clip }}
        >
          {/* Left Section - Text Block */}
          <div className="flex-1 bg-[#55B1FD] p-4 sm:p-6 md:p-8 lg:p-12 xl:p-16 flex flex-col justify-start pt-[7rem] sm:pt-[8rem] md:pt-[9rem] lg:pt-[10rem] xl:pt-[11rem] relative min-h-[50vh] lg:min-h-screen">
            <div className="flex flex-col">
              {/* Contenedores con overflow hidden individual para cada línea */}
              <div
                className="text-black text-left leading-[0.65] sm:leading-[0.7] md:leading-[0.75] lg:leading-[0.65] mb-2 sm:mb-3 md:mb-4"
                style={{ fontFamily: "Formula Condensed", fontWeight: 900 }}
              >
                {/* Primera línea: 03 LIQUID */}
                <div className="overflow-hidden mb-1">
                  <motion.div
                    className="flex flex-col sm:flex-row sm:items-center sm:gap-3 md:gap-4 lg:gap-5 xl:gap-6"
                    style={{ y: title3Y }}
                  >
                    <span
                      className="text-white text-3xl sm:text-4xl md:text-6xl lg:text-7xl xl:text-8xl 2xl:text-9xl 3xl:text-10xl"
                      style={{
                        fontFamily: "Formula Condensed",
                        fontWeight: 900,
                      }}
                    >
                      03
                    </span>
                    <span
                      className="text-3xl sm:text-4xl md:text-6xl lg:text-7xl xl:text-8xl 2xl:text-9xl 3xl:text-10xl"
                      style={{
                        fontFamily: "Formula Condensed",
                        fontWeight: 900,
                      }}
                    >
                      EXECUTE.
                    </span>
                  </motion.div>
                </div>

                {/* Segunda línea: GAMING */}
                <div className="overflow-hidden mb-1">
                  <motion.div
                    className="text-3xl sm:text-4xl md:text-6xl lg:text-7xl xl:text-8xl 2xl:text-9xl 3xl:text-10xl"
                    style={{
                      fontFamily: "Formula Condensed",
                      fontWeight: 900,
                      y: title3Y,
                    }}
                  >
                    WIN.
                  </motion.div>
                </div>

                {/* Tercera línea: ECONOMY */}
                <div className="overflow-hidden">
                  <motion.div
                    className="text-3xl sm:text-4xl md:text-6xl lg:text-7xl xl:text-8xl 2xl:text-9xl 3xl:text-10xl"
                    style={{
                      fontFamily: "Formula Condensed",
                      fontWeight: 900,
                      y: title3Y,
                    }}
                  >
                    REPEAT.
                  </motion.div>
                </div>
              </div>

              <p
                className="text-black text-sm sm:text-base md:text-lg lg:text-xl max-w-full sm:max-w-lg md:max-w-xl lg:max-w-2xl mt-4 sm:mt-6 md:mt-8 leading-relaxed"
                style={{ fontFamily: "Helvetica Now Text", fontWeight: 400 }}
              >
                You get the intel. You make the move. You stack the wins.
$TECH delivers raw signal, not recycled hype — actionable, instant, and ahead of the crowd.
Click in, copy conviction, and repeat until it's second nature.
              </p>
            </div>

            <div className="absolute bottom-4 sm:bottom-6 md:bottom-8 lg:bottom-12 xl:bottom-16 left-4 sm:left-6 md:left-8 lg:left-12 xl:left-16 text-black text-xs font-medium">
              3/4
            </div>
          </div>

          {/* Right Section - Image Block */}
          <div className="flex-1 p-4 sm:p-6 md:p-8 lg:p-12 xl:p-16 flex items-center justify-center bg-[#55B1FD] min-h-[50vh] lg:min-h-screen">
            <div className="w-full max-w-md sm:max-w-lg md:max-w-xl lg:max-w-2xl xl:max-w-5xl h-[31vh] sm:h-[38vh] md:h-[47vh] lg:h-[54vh] xl:h-[62vh] rounded-2xl sm:rounded-3xl overflow-hidden">
              <img
                src="/image3.png"
                alt="Card 3 Image"
                className="w-full h-full object-cover"
              />
            </div>
          </div>
        </motion.section>

        <motion.section
          className="h-screen w-full flex flex-col lg:flex-row sticky top-0"
          style={{ clipPath: card4Clip }}
        >
          {/* Carta 4 - Totalmente blanca vacía */}
          <div className="w-full h-screen bg-white flex items-center justify-center">
            {/* Contenido vacío - solo fondo blanco */}
          </div>
        </motion.section>
      </article>

      {/* Sección HOW IT WORKS - Con animación slide-in desde la izquierda */}
      <section
        ref={howItWorksRef}
        className="h-screen w-full bg-white relative overflow-hidden"
      >
        {/* Título HOW IT WORKS - Posición absoluta en la esquina inferior izquierda */}
        <motion.div
          className="absolute bottom-4 md:bottom-8 lg:bottom-12 xl:bottom-16 left-4 md:left-8 lg:left-12 xl:left-16 z-10 scale-[0.5] sm:scale-[0.6] md:scale-[0.7] lg:scale-[0.8] xl:scale-[0.9] 2xl:scale-[1.0] origin-bottom-left"
          initial={{ x: "-100%", opacity: 0 }}
          animate={
            isInView ? { x: "0%", opacity: 1 } : { x: "-100%", opacity: 0 }
          }
          transition={{
            duration: 1.2,
            ease: [0.25, 0.1, 0.25, 1],
            delay: 0.3,
          }}
        >
          <h2
            className="text-black text-[9rem] font-black leading-none"
            style={{ fontFamily: "Formula Condensed", fontWeight: 900 }}
          >
            HOW IT
            <br />
            WORKS
          </h2>
        </motion.div>

        {/* Contenedor del orbital - Centrado en toda la pantalla + responsivo */}
        <motion.div
          className="absolute inset-0 flex items-center justify-center transform translate-x-[15%] scale-[0.4] sm:scale-[0.5] md:scale-[0.65] lg:scale-[0.8] xl:scale-[0.9] 2xl:scale-[1.0]"
          style={{ scale: orbitalScale }}
        >
          {/* Órbita circular con iconos - tamaño fijo perfecto de 1920x1080 */}
          <div className="relative w-[48rem] h-[48rem]">
            {/* Círculo de órbita (línea punteada) */}
            <div className="absolute inset-0 border-2 border-dashed border-gray-300 rounded-full"></div>

            {/* Logo 3D en el centro - tamaño fijo perfecto de 1920x1080 */}
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[57.6rem] h-[57.6rem] flex items-center justify-center">
              <img
                src="/logo3D.png"
                alt="Logo 3D"
                className="w-full h-full object-contain"
              />
            </div>

            {/* Iconos distribuidos en la órbita - AGRANDADOS 15% */}
            {/* Icono superior - información */}
            <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[2.3rem] h-[2.3rem] bg-pink-500 rounded-full flex items-center justify-center">
              <svg
                className="w-[1.15rem] h-[1.15rem] text-white"
                fill="currentColor"
                viewBox="0 0 24 24"
              >
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z" />
              </svg>
            </div>

            {/* Icono derecha superior - gaming */}
            <div className="absolute top-6 right-0 transform translate-x-[-15px] -translate-y-[-200px] w-[2.3rem] h-[2.3rem] bg-purple-600 rounded-full flex items-center justify-center">
              <svg
                className="w-[1.15rem] h-[1.15rem] text-white"
                fill="currentColor"
                viewBox="0 0 24 24"
              >
                <path d="M17.622 3.027C16.5 2.404 15.155 2 13.7 2c-2.76 0-5.1 1.8-6.4 4.3-.37.71-.6 1.51-.6 2.36 0 1.71.8 3.24 2.1 4.27v8.09c0 .83.67 1.5 1.5 1.5h.5c.83 0 1.5-.67 1.5-1.5v-7.59c.33.05.67.08 1 .08s.67-.03 1-.08v7.59c0 .83.67 1.5 1.5 1.5h.5c.83 0 1.5-.67 1.5-1.5v-8.09c1.3-1.03 2.1-2.56 2.1-4.27 0-.85-.23-1.65-.6-2.36-.13-.23-.27-.45-.42-.66z" />
              </svg>
            </div>

            {/* Icono derecha inferior - tienda */}
            <div className="absolute bottom-[20%] right-0 transform translate-x-[-50px] w-[2.3rem] h-[2.3rem] bg-purple-600 rounded-full flex items-center justify-center">
              <svg
                className="w-[1.15rem] h-[1.15rem] text-white"
                fill="currentColor"
                viewBox="0 0 24 24"
              >
                <path d="M19 7h-3V6c0-2.21-1.79-4-4-4S8 3.79 8 6v1H5c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V9c0-1.1-.9-2-2-2zM10 6c0-1.1.9-2 2-2s2 .9 2 2v1h-4V6zm8 13H6V9h2v1c0 .55.45 1 1 1s1-.45 1-1V9h4v1c0 .55.45 1 1 1s1-.45 1-1V9h2v10z" />
              </svg>
            </div>

            {/* Icono inferior - base de datos */}
            <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 translate-y-1/2 w-[2.3rem] h-[2.3rem] bg-purple-600 rounded-full flex items-center justify-center">
              <svg
                className="w-[1.15rem] h-[1.15rem] text-white"
                fill="currentColor"
                viewBox="0 0 24 24"
              >
                <path d="M12 3C7.58 3 4 4.79 4 7s3.58 4 8 4 8-1.79 8-4-3.58-4-8-4zM4 9v3c0 2.21 3.58 4 8 4s8-1.79 8-4V9c0 2.21-3.58 4-8 4s-8-1.79-8-4zm0 5v3c0 2.21 3.58 4 8 4s8-1.79 8-4v-3c0 2.21-3.58 4-8 4s-8-1.79-8-4z" />
              </svg>
            </div>

            {/* Icono izquierda inferior - información */}
            <div className="absolute bottom-6 left-0 transform translate-x-[50px] translate-y-[-130px]  w-[2.3rem] h-[2.3rem] bg-pink-500 rounded-full flex items-center justify-center">
              <svg
                className="w-[1.15rem] h-[1.15rem] text-white"
                fill="currentColor"
                viewBox="0 0 24 24"
              >
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z" />
              </svg>
            </div>

            {/* Icono izquierda superior - información */}
            <div className="absolute top-[20%] left-0 transform translate-x-[15px] -translate-y-[-60px] w-[2.3rem] h-[2.3rem] bg-pink-500 rounded-full flex items-center justify-center">
              <svg
                className="w-[1.15rem] h-[1.15rem] text-white"
                fill="currentColor"
                viewBox="0 0 24 24"
              >
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z" />
              </svg>
            </div>
          </div>
        </motion.div>
      </section>
    </main>
  );
}
