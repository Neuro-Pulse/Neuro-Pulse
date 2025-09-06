"use client";

import { cn } from "@/lib/utils";
import { ReactNode, useEffect, useState } from "react";
import {
  motion,
  useScroll,
  useTransform,
  useMotionValueEvent,
} from "framer-motion";
import { usePumpFunLink } from "@/hooks/usePumpFunLink";

interface DecorativeUIFrameProps {
  children: ReactNode;
  className?: string;
}

export const DecorativeUIFrame = ({
  children,
  className,
}: DecorativeUIFrameProps) => {
  const [slashCount, setSlashCount] = useState(20);
  const [menuOpen, setMenuOpen] = useState(false);
  const [currentSection, setCurrentSection] = useState(0);
  const [sectionProgress, setSectionProgress] = useState(0);
  const [globalProgress, setGlobalProgress] = useState(0);
  const { scrollY } = useScroll();
  
  // Hook para obtener el link dinámico de Pump.fun
  const { pumpFunLink } = usePumpFunLink();

  // Configuración de breakpoints en vh - FÁCIL DE MODIFICAR
  const colorBreakpoints = [
    { vh: 0, color: "white" }, // Inicio
    { vh: 30, color: "white" }, // Inicio
    { vh: 75, color: "black" }, // 150vh - cambio a negro
    // { vh: 275, color: "white" },    // 150vh - cambio a negro
    { vh: 500, color: "white" },
    { vh: 600, color: "white" },


    // { vh: 600, color: "black" },
    // { vh: 800, color: "black" },
    // { vh: 830, color: "black" },
    // { vh: 930, color: "black" },
    // { vh: 950, color: "black" },
    { vh: 1065, color: "black" },
    { vh: 1165, color: "white" },
    { vh: 1245, color: "black" }, // Cambio a negro
    { vh: 1515, color: "black" }, // Cambio a negro
    { vh: 1555, color: "white" },

    // 150vh - cambio a negro
  ];

  // Configuración específica para links e iconos - SOLO NEGRO Y BLANCO
  const linkColorBreakpoints = [
    { vh: 0, color: "white" }, // Inicio
    { vh: 100, color: "white" }, // Mantener negro
    { vh: 100, color: "black" }, // Cambio a blanco
    { vh: 700, color: "black" }, // Mantener blanco
    { vh: 865, color: "black" }, // Mantener blanco
    { vh: 865, color: "black" }, // Cambio a negro
    { vh: 900, color: "black" }, // Cambio a negro
    { vh: 950, color: "black" }, // Cambio a negro
    { vh: 1165, color: "black" }, // Cambio a negro
    { vh: 1165, color: "white" }, // Cambio a negro
    { vh: 1245, color: "white" }, // Cambio a negro
    { vh: 1245, color: "black" }, // Cambio a negro
    { vh: 1515, color: "black" }, // Cambio a negro
    { vh: 1555, color: "white" }, // Cambio a negro
  ];

  // Convierte vh a píxeles
  const vhToPixels = (vh: number) => {
    if (typeof window !== "undefined") {
      return (vh * window.innerHeight) / 100;
    }
    return vh * 10; // fallback
  };

  // Extrae los valores de vh y colores
  const vhValues = colorBreakpoints.map((bp) => vhToPixels(bp.vh));
  const frameColors = colorBreakpoints.map((bp) =>
    bp.color === "white" ? "rgb(255, 255, 255)" : "rgb(0, 0, 0)"
  );
  const textColors = colorBreakpoints.map((bp) =>
    bp.color === "white" ? "rgb(255, 255, 255)" : "rgb(0, 0, 0)"
  );

  // Valores específicos para links e iconos
  const linkVhValues = linkColorBreakpoints.map((bp) => vhToPixels(bp.vh));
  const linkColors = linkColorBreakpoints.map((bp) =>
    bp.color === "white" ? "rgb(255, 255, 255)" : "rgb(0, 0, 0)"
  );
  const linkBgColors = linkColorBreakpoints.map(
    (bp) => (bp.color === "white" ? "rgb(0, 0, 0)" : "rgb(255, 255, 255)") // Contrario para fondo del botón
  );

  // Colores invertidos para el div de la imagen del botón
  const linkInvertedColors = linkColorBreakpoints.map(
    (bp) => (bp.color === "white" ? "rgb(0, 0, 0)" : "rgb(255, 255, 255)") // Invertido: white -> black, black -> white
  );

  // Filtro para el logo - invierte cuando linkColorBreakpoints es "black"
  // Logo blanco -> se invierte a negro cuando bp.color es "black"
  const logoFilters = linkColorBreakpoints.map(
    (bp) => (bp.color === "black" ? "invert(1)" : "invert(0)") // Invierte el logo blanco a negro
  );

  // Filtro específico para logo verde -> blanco
  // Convierte verde a blanco cuando bp.color es "black" (invertido)
  const logoGreenToWhiteFilters = linkColorBreakpoints.map(
    (bp) =>
      bp.color === "black"
        ? "brightness(100) saturate(0) contrast(0) brightness(2)"
        : "none" // Verde a blanco cuando es "black"
  );

  const frameColor = useTransform(scrollY, vhValues, frameColors);
  const textColor = useTransform(scrollY, vhValues, textColors);

  // Transforms específicos para links e iconos
  const linkTextColor = useTransform(scrollY, linkVhValues, linkColors);
  const linkBgColor = useTransform(scrollY, linkVhValues, linkBgColors);
  const linkInvertedColor = useTransform(
    scrollY,
    linkVhValues,
    linkInvertedColors
  );
  const logoFilter = useTransform(scrollY, linkVhValues, logoFilters);
  const logoGreenToWhiteFilter = useTransform(
    scrollY,
    linkVhValues,
    logoGreenToWhiteFilters
  );

  // Configuración de secciones - heights en vh
  const sectionHeights = [
    { start: 0, end: 100 }, // HeroSection
    { start: 100, end: 200 }, // VelocityScroll
    { start: 200, end: 500 }, // ParallaxLayers
    { start: 500, end: 800 }, // HorizontalScroll
    { start: 800, end: 950 }, // Better For All section
  ];

  // Calcula el progreso global
  useMotionValueEvent(scrollY, "change", (latest) => {
    // Calcula el progreso global del scroll
    const totalScrollHeight = document.body.scrollHeight - window.innerHeight;
    const globalScrollPercent = Math.max(
      0,
      Math.min(100, (latest / totalScrollHeight) * 100)
    );
    setGlobalProgress(globalScrollPercent);
  });

  const calculateSlashes = () => {
    const width = window.innerWidth;
    let count;
    if (width < 640) {
      count = Math.floor(width / 11);
    } else if (width < 768) {
      count = Math.floor(width / 16);
    } else {
      count = Math.floor(width / 21.5);
    }
    setSlashCount(count);
  };

  useEffect(() => {
    calculateSlashes();
    window.addEventListener("resize", calculateSlashes);
    return () => window.removeEventListener("resize", calculateSlashes);
  }, []);

  return (
    <div className="relative max-h-screen overflow-hidden w-full pointer-events-none">
      {/* Main Border Frame */}
      <motion.div
        className={cn(
          "fixed z-50 pointer-events-none border",
          "inset-1 rounded-[8px] sm:inset-2 sm:rounded-[12px] md:inset-3 md:rounded-[16px]"
        )}
        style={{ borderColor: frameColor }}
      />

      {/* Top Horizontal Line */}
      <div className="fixed z-30 pointer-events-none top-[40px] left-1 right-1 sm:top-[50px] sm:left-2 sm:right-2 md:top-[60px] md:left-3 md:right-3">
        <motion.div
          className="h-px w-full translate-y-[25px] sm:translate-y-[35px] md:translate-y-[45px]"
          style={{ backgroundColor: frameColor }}
        />
      </div>

      {/* Bottom Horizontal Line */}
      <div className="fixed z-30 pointer-events-none bottom-[40px] left-1 right-1 sm:bottom-[50px] sm:left-2 sm:right-2 md:bottom-[60px] md:left-3 md:right-3">
        <motion.div
          className="h-px w-full translate-y-[-35px] sm:translate-y-[-55px] md:translate-y-[-65px]"
          style={{ backgroundColor: frameColor }}
        />
      </div>

      {/* Content Area */}
      <div
        className={cn(
          "relative z-20 min-h-screen p-6 m-3 pt-8 pb-16",
          className
        )}
      >
        {children}
      </div>

      {/* MOBILE MENU SYSTEM - Simplified with maximum z-index */}
      {/* Hamburger Button */}
      <div 
        className="md:hidden fixed top-6 right-6 pointer-events-auto" 
        style={{ zIndex: 999999999 }}
      >
        <button
          className="focus:outline-none p-3 bg-black/20 rounded-md hover:bg-black/40 transition-colors duration-200"
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log("Hamburguesa clickeada!", !menuOpen);
            setMenuOpen(!menuOpen);
          }}
          style={{ zIndex: 999999999 }}
        >
          <div
            className={`w-6 h-0.5 bg-white my-1.5 transition-all duration-300 ${
              menuOpen ? "rotate-45 translate-y-2" : ""
            }`}
          ></div>
          <div
            className={`w-6 h-0.5 bg-white my-1.5 transition-all duration-300 ${
              menuOpen ? "opacity-0" : ""
            }`}
          ></div>
          <div
            className={`w-6 h-0.5 bg-white my-1.5 transition-all duration-300 ${
              menuOpen ? "-rotate-45 -translate-y-2" : ""
            }`}
          ></div>
        </button>
      </div>

      {/* Mobile Menu Overlay */}
      {menuOpen && (
        <div
          className="fixed top-0 left-0 w-full h-full bg-black/95 flex flex-col items-center justify-center"
          style={{ zIndex: 999999998 }}
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            setMenuOpen(false);
          }}
        >
          <div 
            className="flex flex-col items-center gap-8 pointer-events-auto"
            onClick={(e) => e.stopPropagation()}
            style={{ zIndex: 999999999 }}
          >
            {/* Botones sin enlaces externos */}
            <div
              className="text-2xl font-helvetica-now-text font-bold uppercase text-white hover:text-[#CDFE00] transition-colors duration-300 cursor-pointer"
              style={{ letterSpacing: "0.25em" }}
              onClick={() => setMenuOpen(false)}
            >
              PUMP.FUN
            </div>

            <div
              className="text-2xl font-helvetica-now-text font-bold uppercase text-white hover:text-[#CDFE00] transition-colors duration-300 cursor-pointer"
              style={{ letterSpacing: "0.25em" }}
              onClick={() => setMenuOpen(false)}
            >
              EXTENSION
            </div>

            <div
              className="text-2xl font-helvetica-now-text font-bold uppercase text-white hover:text-[#CDFE00] transition-colors duration-300 cursor-pointer"
              style={{ letterSpacing: "0.25em" }}
              onClick={() => setMenuOpen(false)}
            >
              WHITEPAPER
            </div>

            <div
              className="text-2xl font-helvetica-now-text font-bold uppercase text-white hover:text-[#CDFE00] transition-colors duration-300 cursor-pointer"
              style={{ letterSpacing: "0.25em" }}
              onClick={() => setMenuOpen(false)}
            >
              DEVELOPER
            </div>

            <div
              className="flex items-center bg-white rounded-[20px] overflow-hidden mt-4 cursor-pointer"
              onClick={() => setMenuOpen(false)}
            >
              <span
                className="px-6 py-3 text-lg font-helvetica-now-text font-bold uppercase text-black"
                style={{ letterSpacing: "0.25em" }}
              >
                Buy $NEURO
              </span>
              <div className="bg-black px-4 py-3 flex items-center justify-center rounded-l-lg">
                <img
                  className="scale-[2]"
                  src="logoWhite.png"
                  width={40}
                  height={40}
                  alt="logo"
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Bottom Slash Line */}
      <div className="fixed bottom-2 left-0 right-0 z-30 pointer-events-none sm:bottom-3 md:bottom-4">
        <div className="flex justify-center items-center h-4 sm:h-6 md:h-8 w-full">
          <motion.div
            className="text-[12px] sm:text-lg md:text-xl font-bb-mono tracking-wider text-center w-full overflow-hidden"
            style={{ color: frameColor }}
          >
            {Array.from({ length: slashCount }, (_, i) => (
              <span
                key={i}
                className="inline-block mr-0.5 sm:mr-0.5 md:mr-2 font-bb-mono"
              >
                /
              </span>
            ))}
          </motion.div>
        </div>
      </div>

      {/* Logo */}
      <div className="fixed top-1 left-6 z-50 flex items-center gap-3">
        <motion.img
          src="logoWhite.png"
          alt="Logo"
          className="w-16 sm:w-20 md:w-28"
          style={{
            filter: logoFilter,
            transition: "filter 0.6s cubic-bezier(0.4, 0, 0.2, 1)",
          }}
        />
        <motion.span
          className="text-2xl sm:text-3xl md:text-4xl font-bold"
          style={{
            fontFamily: "Formula Condensed, sans-serif",
            fontWeight: 900,
            color: linkTextColor,
            transform: "scale(1.125) translateY(2px)",
            letterSpacing: "2.5px",
          }}
        >
          Neuro Pulse
        </motion.span>
      </div>

      {/* Navigation Buttons */}
      <div className="hidden md:flex fixed top-8 right-8 z-50 items-center gap-10 pointer-events-auto">
        {/* Botones sin enlaces externos */}
        <motion.div
          className="text-lg font-helvetica-now-text font-bold uppercase hover:text-[#CDFE00] hover:scale-105 transition-all duration-300 ease-out cursor-pointer relative group"
          style={{ color: linkTextColor }}
        >
          <span className="relative z-10">Pump.fun</span>
          <div className="absolute inset-0 bg-white/5 rounded-lg scale-0 group-hover:scale-100 transition-transform duration-300 ease-out" />
        </motion.div>
        
        <motion.div
          className="text-lg font-helvetica-now-text font-bold uppercase hover:text-[#CDFE00] hover:scale-105 transition-all duration-300 ease-out cursor-pointer relative group"
          style={{ color: linkTextColor }}
        >
          <span className="relative z-10">Extension</span>
          <div className="absolute inset-0 bg-white/5 rounded-lg scale-0 group-hover:scale-100 transition-transform duration-300 ease-out" />
        </motion.div>
        
        <motion.div
          className="text-lg font-helvetica-now-text font-bold uppercase hover:text-[#CDFE00] hover:scale-105 transition-all duration-300 ease-out cursor-pointer relative group"
          style={{ color: linkTextColor }}
        >
          <span className="relative z-10">Whitepaper</span>
          <div className="absolute inset-0 bg-white/5 rounded-lg scale-0 group-hover:scale-100 transition-transform duration-300 ease-out" />
        </motion.div>
        
        <motion.div
          className="text-lg font-helvetica-now-text font-bold uppercase hover:text-[#CDFE00] hover:scale-105 transition-all duration-300 ease-out cursor-pointer relative group"
          style={{ color: linkTextColor }}
        >
          <span className="relative z-10">Developer</span>
          <div className="absolute inset-0 bg-white/5 rounded-lg scale-0 group-hover:scale-100 transition-transform duration-300 ease-out" />
        </motion.div>

        {/* Buy Button */}
        <motion.div
          className="flex items-center rounded-[20px] overflow-hidden hover:scale-105 hover:shadow-lg hover:shadow-[#CDFE00]/20 transition-all duration-300 ease-out cursor-pointer group"
          style={{ backgroundColor: linkTextColor }}
        >
          <motion.span
            className="px-4 py-2 text-lg font-helvetica-now-text font-bold uppercase group-hover:text-[#CDFE00] transition-colors duration-300"
            style={{
              color: linkBgColor,
            }}
          >
            Buy $NEURO
          </motion.span>
          <div
            className="px-3 py-2 flex items-center justify-center transition-all duration-300 rounded-l-lg"
            style={{ backgroundColor: "#000000" }}
          >
            <motion.img
              className="scale-[2.5]"
              src="logo.png"
              width={50}
              height={50}
              alt="logo"
              style={{
                filter: logoGreenToWhiteFilter,
                transition: "filter 0.6s cubic-bezier(0.4, 0, 0.2, 1)",
              }}
            />
          </div>
        </motion.div>

        {/* Global Progress Percentage */}
        <motion.div
          className="fixed top-[120px] right-8 z-50 flex items-center gap-1 pointer-events-none"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <div className="flex items-center gap-0">
            <motion.span
              className="text-sm font-bb-manual-mono-pro-original-semi-regular"
              style={{ color: linkTextColor }}
            >
              +
            </motion.span>
            <motion.span
              className="text-sm font-bb-manual-mono-pro-original-semi-regular px-5"
              style={{ color: linkTextColor }}
            >
              SYS CALC {Math.round(globalProgress)}%
            </motion.span>
            <motion.span
              className="text-sm font-bb-manual-mono-pro-original-semi-regular"
              style={{ color: linkTextColor }}
            >
              +
            </motion.span>
          </div>
        </motion.div>
      </div>

      {/* Social Icons eliminados */}

      {/* Bottom Text Info */}
      <div className="fixed bottom-8 left-0 right-0 sm:bottom-10 md:bottom-12 z-30 pointer-events-none">
        <div className="flex justify-between items-center translate-y-[-20px] px-2 sm:px-4 md:px-6">
          <motion.div
            className="flex justify-start items-center w-1/2 text-[9px] sm:text-xs md:text-sm font-bb-mono gap-[150px]"
            style={{ color: frameColor }}
          >
            <span>*HTM.STS</span>
            <span>VER/12433.56</span>
            <span className="flex flex-col leading-tight translate-y-[10px]">
              <span>RUNTIME.LIVE</span>
              <span>RELATIVE.RUN</span>
            </span>
          </motion.div>
          <div className="w-32 sm:w-36 md:w-40"></div>
        </div>
      </div>
    </div>
  );
};

// Dummy social icon functions (replace with real ones)
function social1() {
  return (
    <svg className="w-10 h-10" fill="currentColor" viewBox="0 0 24 24">
      <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
    </svg>
  );
}
function social2() {
  return (
    <svg className="w-10 h-10" fill="currentColor" viewBox="0 0 24 24">
      <path d="M21.053 20.235c-1.132 1.046-2.636 1.765-4.323 1.765-3.477 0-6.297-2.82-6.297-6.297 0-3.477 2.82-6.297 6.297-6.297 1.687 0 3.191.719 4.323 1.765l.965-.965C20.604 8.82 18.748 7.703 16.73 7.703c-4.582 0-8.297 3.715-8.297 8.297s3.715 8.297 8.297 8.297c2.018 0 3.874-1.117 5.288-2.553l-.965-.965z" />
    </svg>
  );
}
function social3() {
  return (
    <svg className="w-10 h-10" fill="currentColor" viewBox="0 0 24 24">
      <path d="M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm5.894 8.221l-1.97 9.28c-.145.658-.537.818-1.084.508l-3-2.21-1.446 1.394c-.14.18-.357.223-.548.223-.324 0-.454-.157-.454-.454v-4.63L7.37 10.4c-.66-.185-.68-.66.136-.994l11.86-4.573c.548-.202 1.028.126.846.94z" />
    </svg>
  );
}
