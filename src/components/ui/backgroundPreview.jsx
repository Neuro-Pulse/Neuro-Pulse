import React from "react";
import { motion } from "framer-motion";
import { BackgroundBeamsWithCollision } from "@/components/ui/background-beams-with-collision";
import Image from "next/image";
import Head from "next/head";

// Variantes de animación
const fadeInUp = {
  hidden: { opacity: 0, y: 50 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.8, ease: "easeOut" } },
};

export function BackgroundPreview() {
  return (
    <BackgroundBeamsWithCollision>
      <Head>
        <link rel="preload" href="/videoBen.webm" as="video" />
      </Head>

      {/* Contenedor animado */}
      <motion.div
        initial="hidden"
        whileInView="visible"
        viewport={{ once: false, amount: 0.4 }}
        className="flex flex-col-reverse md:flex-row items-center justify-center w-full min-h-screen gap-6 p-4 md:p-12 max-sm:flex-col max-sm:gap-[80px]"
      >
        {/* Sección de la Imagen + Texto */}
        <motion.div
          variants={fadeInUp}
          className="flex flex-col items-center justify-center w-full md:w-1/2 max-sm:order-1"
        >
          <motion.h2
            variants={fadeInUp}
            className="text-2xl relative z-20 md:text-4xl lg:text-7xl font-bold text-center text-black dark:text-white font-sans tracking-tight max-sm:text-4xl"
          >
            <div className="max-sm:block max-sm:mb-4 text-4xl text-start ml-[5%] mb-[-5%] tracking-widest max-sm:mb-[-10px]">
              Introducing
            </div>
            <div className="relative mx-auto inline-block w-max [filter:drop-shadow(0px_1px_3px_rgba(27,_37,_80,_0.14))]">
              <motion.div
                variants={fadeInUp}
                className="absolute left-0 top-[1px] bg-clip-text bg-no-repeat text-transparent bg-gradient-to-r py-4 from-green-500 via-green-200 to-red-500 [text-shadow:0_0_rgba(0,0,0,0.1)] animate-gradient max-sm:text-6xl text-[8.5vw] max-sm:text-[90px] tracking-[2px]"
              >
                <span>$BENE</span>
              </motion.div>
              <motion.div
                variants={fadeInUp}
                className="relative bg-clip-text text-transparent bg-no-repeat bg-gradient-to-r from-green-700 via-white to-red-700 py-4 animate-gradient max-sm:text-6xl  tracking-[2px] text-[8.5vw] max-sm:text-[90px]"
              >
                <span>$BENE</span>
              </motion.div>
            </div>
            <motion.div
              variants={fadeInUp}
              className="flex items-center w-full -space-x-2  mt-[-20px] max-sm:mt-[-65px] h-full"
            >
              <div className="w-12 h-12 relative max-sm:mt-[45px]">
                <Image
                  src="/solana.png"
                  alt="solana"
                  fill
                  className="object-cover rounded-full"
                />
              </div>
              <p className="text-start text-sm md:text-[18px] bg-clip-text text-transparent bg-gradient-to-r from-blue-200 to-blue-400 whitespace-nowrap tracking-[2px] max-sm:mb-[-50px]">
                Powered By Solana
              </p>
            </motion.div>
          </motion.h2>
        </motion.div>

        {/* Sección del Video */}
        <motion.div
          variants={fadeInUp}
          className="z-[2] flex flex-col items-center justify-center w-full md:w-1/2 text-center max-sm:order-2"
        >
          <motion.video
            variants={fadeInUp}
            src="/videoBen.webm"
            autoPlay
            loop
            muted
            playsInline
            preload="auto"
            className="w-full max-w-xs md:max-w-md rounded-lg shadow-lg animated-border"
          />
          <motion.p
            variants={fadeInUp}
            className="mt-4 text-lg text-gray-800 dark:text-gray-300 mx-[25%] mt-[50px] max-sm:mx-auto"
          >
            Let me show you the Italian side of Solana's blockchain
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-green-500 via-white to-red-500">
              {" "}
              $BENE
            </span>{" "}
            not yet launched 🚀.
          </motion.p>
        </motion.div>
      </motion.div>
    </BackgroundBeamsWithCollision>
  );
}
