import { Geist, Geist_Mono } from "next/font/google";
import Head from "next/head";
import { motion } from "framer-motion";
import { useState } from "react";

import { BackgroundPreview } from "@/components/ui/backgroundPreview";
import MainExperience from "@/components/ui/MainExperience";
import { BackgroundGradientAnimationDemo } from "@/components/ui/background-gradient-final";
import { DecorativeUIFrame } from "@/components/ui/decorative-ui-frame";
import { HeroSection } from "@/components/ui/hero-section";
import { VelocityScroll } from "@/components/ui/velocity-scroll";
import ParallaxLayers from "@/components/ui/parallax-layers";
import HorizontalScroll from "@/components/ui/HorizontalScroll";
import { PreLoader } from "@/components/ui/PreLoader";
// import Footer from "@/components/Footer";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export default function Home() {
  const [isLoading, setIsLoading] = useState(true);

  // Variantes de animación para el fade in
  const pageVariants = {
    initial: {
      opacity: 0
    },
    animate: {
      opacity: 1,
      transition: {
        duration: 1.2,
        ease: "easeOut",
        staggerChildren: 0.2
      }
    }
  };

  const childVariants = {
    initial: {
      opacity: 0
    },
    animate: {
      opacity: 1,
      transition: {
        duration: 0.8,
        ease: "easeOut"
      }
    }
  };

  return (
    <>
      {/* PreLoader */}
      {isLoading && (
        <PreLoader onComplete={() => setIsLoading(false)} />
      )}
      
      {/* Main Content */}
      <motion.div
        initial="initial"
        animate={!isLoading ? "animate" : "initial"}
        variants={pageVariants}
        style={{ display: isLoading ? 'none' : 'block' }}
      >
        <Head>
          <title>Neuro Pulse</title>
        </Head>
      {/* Fixed background gradient */}
      {/* <div className="max-h-screen max-w-screen overflow-hidden"> */}
        <motion.div variants={childVariants}>
          <BackgroundGradientAnimationDemo/>
        </motion.div>
      {/* </div> */}
      
      {/* Decorative UI frame with blend-mode */}
      <motion.div variants={childVariants}>
        <DecorativeUIFrame>
            <HeroSection />
        </DecorativeUIFrame>
      </motion.div>

      <motion.div variants={childVariants} className="h-screen w-full bg-white">
        <VelocityScroll />
      </motion.div>
      
      {/* Parallax Layers Effect - Ahora se conecta directamente */}
      <motion.div variants={childVariants}>
        <ParallaxLayers />
      </motion.div>
{/*       
      <div className="h-screen w-full">
      </div>
      <div className="h-screen w-full">
      </div> */}
      
      {/* <BackgroundPreview /> */}
      {/* <MainExperience /> */}
      {/* <div className="bg-blue-500"></div> */}

      {/* Horizontal Scroll Section */}
      <motion.div variants={childVariants}>
        <HorizontalScroll />
      </motion.div>
      
      {/* Main content section that sits on top of the footer */}
      <motion.div 
        variants={childVariants}
        className="relative w-full h-[150vh] z-[5] bg-black text-white overflow-hidden"
      >
        {/* Layout principal - Responsive: vertical para móvil/tablet, horizontal para desktop */}
        <div className="h-full w-full flex flex-col lg:flex-row">
          
          {/* MOBILE/TABLET LAYOUT (hasta lg) */}
          {/* 1. BETTER FOR - Primer elemento en móvil */}
          <div className="lg:hidden flex flex-col items-center justify-center py-8 px-6 order-1">
            <h1 className="text-white text-[4rem] sm:text-[6rem] md:text-[8rem] font-black leading-[0.9] text-center" style={{ fontFamily: 'Formula Condensed', fontWeight: 900 }}>
              BETTER
            </h1>
            <h1 className="text-white text-[4rem] sm:text-[6rem] md:text-[8rem] font-black leading-[0.8] text-center" style={{ fontFamily: 'Formula Condensed', fontWeight: 900 }}>
              FOR
            </h1>
          </div>
          
          {/* 2. IMAGEN - Segundo elemento en móvil */}
          <div className="lg:hidden flex items-center justify-center py-8 order-2">
            <img 
              src="/logo3Dsecond.png" 
              alt="Logo 3D" 
              className="w-[24rem] h-[24rem] sm:w-[30rem] sm:h-[30rem] md:w-[36rem] md:h-[36rem] object-contain"
            />
          </div>
          
          {/* 3. ALL - Tercer elemento en móvil */}
          <div className="lg:hidden flex items-center justify-center py-4 order-3">
            <h1 className="text-white text-[4rem] sm:text-[6rem] md:text-[8rem] font-black leading-[0.8] text-center" style={{ fontFamily: 'Formula Condensed', fontWeight: 900 }}>
              ALL
            </h1>
          </div>
          
          {/* 4. NAVIGATION - Cuarto elemento en móvil */}
          <div className="lg:hidden flex flex-col items-center py-8 px-6 order-4">
            <h2 className="text-white text-base font-light mb-6 text-center" style={{ fontFamily: 'Formula Condensed', fontWeight: 300 }}>
              ▶ NAVIGATION
            </h2>
            <nav className="space-y-4 text-center">
              <div 
                className="text-white text-xl font-light cursor-pointer hover:text-gray-300 transition-colors" 
                style={{ fontFamily: 'Formula Condensed', fontWeight: 300 }}
                onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
              >
                Home
              </div>
              <div 
                className="text-white text-xl font-light cursor-pointer hover:text-gray-300 transition-colors" 
                style={{ fontFamily: 'Formula Condensed', fontWeight: 300 }}
              >
                Whitepaper
              </div>
              <div 
                className="text-white text-xl font-light cursor-pointer hover:text-gray-300 transition-colors" 
                style={{ fontFamily: 'Formula Condensed', fontWeight: 300 }}
              >
                Twitter
              </div>
              <div 
                className="text-white text-xl font-light cursor-pointer hover:text-gray-300 transition-colors" 
                style={{ fontFamily: 'Formula Condensed', fontWeight: 300 }}
              >
                GitHub
              </div>
            </nav>
          </div>
          
          {/* DESKTOP LAYOUT (lg y superior) - Diseño original perfecto */}
          {/* Lado izquierdo - BETTER FOR */}
          <div className="hidden lg:flex flex-1 flex-col justify-center pl-16 translate-y-[300px]">
            <h1 className="text-white text-[10.2rem] xl:text-[17rem] font-black leading-[0.9]" style={{ fontFamily: 'Formula Condensed', fontWeight: 900 }}>
              BETTER
            </h1>
            <h1 className="text-white text-[10.2rem] xl:text-[17rem] font-black leading-[0.8]" style={{ fontFamily: 'Formula Condensed', fontWeight: 900 }}>
              FOR
            </h1>
          </div>
          
          {/* Centro - Logo 3D */}
          <div className="hidden lg:flex flex-1 translate-x-[100px] translate-y-[100px] items-center justify-center overflow-hidden">
            <img 
              src="/logo3Dsecond.png" 
              alt="Logo 3D" 
              className="w-[40rem] xl:w-[50rem] h-[40rem] xl:h-[50rem] object-contain transform scale-[2.5]"
            />
          </div>
          
          {/* Lado derecho - Navigation y ALL */}
          <div className="hidden lg:flex flex-1 flex-col translate-x-[200px] translate-y-[350px] justify-between pr-16 py-16">
            {/* Navigation superior */}
            <div>
              <h2 className="text-white text-base font-light mb-8" style={{ fontFamily: 'Formula Condensed', fontWeight: 300 }}>
                ▶ NAVIGATION
              </h2>
              <nav className="space-y-4">
                <div 
                  className="text-white text-2xl xl:text-3xl font-light cursor-pointer hover:text-gray-300 transition-colors block" 
                  style={{ fontFamily: 'Formula Condensed', fontWeight: 300 }}
                  onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                >
                  Home
                </div>
                <div 
                  className="text-white text-2xl xl:text-3xl font-light cursor-pointer hover:text-gray-300 transition-colors block" 
                  style={{ fontFamily: 'Formula Condensed', fontWeight: 300 }}
                >
                  Whitepaper
                </div>
                <div 
                  className="text-white text-2xl xl:text-3xl font-light cursor-pointer hover:text-gray-300 transition-colors block" 
                  style={{ fontFamily: 'Formula Condensed', fontWeight: 300 }}
                >
                  Twitter
                </div>
                <div 
                  className="text-white text-2xl xl:text-3xl font-light cursor-pointer hover:text-gray-300 transition-colors block" 
                  style={{ fontFamily: 'Formula Condensed', fontWeight: 300 }}
                >
                  GitHub
                </div>
              </nav>
            </div>
            
            {/* ALL inferior */}
            <div className="self-end">
              <h1 className="text-white translate-y-[-450px] translate-x-[-200px] text-[10.2rem] xl:text-[17rem] font-black leading-[0.8]" style={{ fontFamily: 'Formula Condensed', fontWeight: 900 }}>
                ALL
              </h1>
            </div>
          </div>
        </div>
        
        {/* Información inferior */}
        <div className="absolute bottom-8 md:bottom-16 left-8 md:left-16 right-8 md:right-16 flex justify-between items-end">
          {/* JOIN THE FUTURE */}
          <div>
            <h3 className="text-white text-sm md:text-base font-light mb-4" style={{ fontFamily: 'Formula Condensed', fontWeight: 300 }}>
              ▶ JOIN THE FUTURE
            </h3>
          </div>
          
          {/* Información legal y contacto */}
          <div className="flex gap-16 md:gap-24">
            <div className="space-y-2">
             
              
            </div>
            
            <div className="space-y-2">
              
             
            </div>
          </div>
        </div>
      </motion.div>
      
      {/* Footer Section */}
      {/* <motion.div variants={childVariants}>
        <Footer />
      </motion.div> */}
      </motion.div>
    </>
  );
}
