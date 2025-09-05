"use client";
import { cn } from "@/lib/utils";
import React, { useEffect, useRef, useState, ReactNode } from "react";

interface BackgroundGradientAnimationProps {
  gradientBackgroundStart?: string;
  gradientBackgroundEnd?: string;
  firstColor?: string;
  secondColor?: string;
  thirdColor?: string;
  fourthColor?: string;
  fifthColor?: string;
  pointerColor?: string;
  size?: string;
  blendingValue?: string;
  children?: ReactNode;
  className?: string;
  interactive?: boolean;
  containerClassName?: string;
}

const palettes = [
  {
    name: 'Volcanic Rage',
    gradientBackgroundStart: 'rgb(139, 0, 0)',
    gradientBackgroundEnd: 'rgb(75, 0, 0)',
    colors: [
      '255, 69, 0',     // Naranja rojo volcánico
      '255, 140, 0',    // Naranja brillante
      '255, 69, 0',     // Rojo volcánico consistente
      '220, 20, 60',    // Crimson
      '178, 34, 34',    // Rojo fuego
      '255, 99, 71'     // Tomate ardiente
    ],
    pointerColor: '255, 69, 0'
  },
  {
    name: 'Electric Lime',
    gradientBackgroundStart: 'rgb(0, 100, 0)',
    gradientBackgroundEnd: 'rgb(0, 50, 0)',
    colors: [
      '50, 205, 50',     // Lima brillante
      '124, 252, 0',     // Verde neón
      '0, 255, 0',       // Verde puro
      '173, 255, 47',    // Verde amarillo
      '154, 205, 50',    // Verde oliva
      '0, 255, 127'      // Verde primavera
    ],
    pointerColor: '124, 252, 0'
  },
  {
    name: 'Deep Ocean Electric',
    gradientBackgroundStart: 'rgb(0, 0, 139)',
    gradientBackgroundEnd: 'rgb(0, 0, 80)',
    colors: [
      '0, 191, 255',     // Azul cielo profundo
      '30, 144, 255',    // Azul dodger
      '0, 255, 255',     // Cyan brillante
      '72, 209, 204',    // Turquesa medio
      '0, 206, 209',     // Turquesa oscuro
      '65, 105, 225'     // Azul real
    ],
    pointerColor: '0, 191, 255'
  },
  {
    name: 'Magenta Madness',
    gradientBackgroundStart: 'rgb(139, 0, 139)',
    gradientBackgroundEnd: 'rgb(75, 0, 130)',
    colors: [
      '255, 0, 255',     // Magenta puro
      '255, 20, 147',    // Rosa profundo
      '186, 85, 211',    // Orquídea medio
      '138, 43, 226',    // Violeta azul
      '148, 0, 211',     // Violeta oscuro
      '199, 21, 133'     // Rosa violeta
    ],
    pointerColor: '255, 0, 255'
  },
  {
    name: 'Sunset Explosion',
    gradientBackgroundStart: 'rgb(255, 69, 0)',
    gradientBackgroundEnd: 'rgb(139, 69, 19)',
    colors: [
      '255, 140, 0',     // Naranja dorado
      '255, 165, 0',     // Naranja
      '255, 215, 0',     // Dorado
      '255, 69, 0',      // Rojo naranja
      '255, 99, 71',     // Tomate
      '255, 127, 80'     // Coral
    ],
    pointerColor: '255, 140, 0'
  },
  {
    name: 'Midnight Neon',
    gradientBackgroundStart: 'rgb(25, 25, 25)',
    gradientBackgroundEnd: 'rgb(0, 0, 0)',
    colors: [
      '0, 255, 255',     // Cyan neón
      '255, 0, 255',     // Magenta neón
      '255, 255, 0',     // Amarillo neón
      '57, 255, 20',     // Verde neón
      '255, 105, 180',   // Rosa neón suave (reemplaza el rosa fuerte)
      '138, 43, 226'     // Violeta neón
    ],
    pointerColor: '0, 255, 255'
  },
  {
    name: 'Royal Purple Storm',
    gradientBackgroundStart: 'rgb(72, 61, 139)',
    gradientBackgroundEnd: 'rgb(25, 25, 112)',
    colors: [
      '138, 43, 226',    // Violeta azul
      '147, 112, 219',   // Púrpura claro
      '186, 85, 211',    // Orquídea medio
      '75, 0, 130',      // Índigo
      '148, 0, 211',     // Violeta oscuro
      '123, 104, 238'    // Pizarra azul medio
    ],
    pointerColor: '138, 43, 226'
  },
  {
    name: 'Emerald Matrix',
    gradientBackgroundStart: 'rgb(0, 100, 0)',
    gradientBackgroundEnd: 'rgb(0, 128, 0)',
    colors: [
      '0, 255, 0',       // Verde puro
      '50, 205, 50',     // Verde lima
      '124, 252, 0',     // Verde césped
      '0, 250, 154',     // Verde primavera medio
      '102, 205, 170',   // Verde mar medio
      '60, 179, 113'     // Verde mar
    ],
    pointerColor: '0, 255, 0'
  },
  {
    name: 'Crimson Lightning',
    gradientBackgroundStart: 'rgb(220, 20, 60)',
    gradientBackgroundEnd: 'rgb(139, 0, 0)',
    colors: [
      '220, 20, 60',     // Crimson (pointerColor también)
      '255, 69, 0',      // Rojo naranja
      '255, 0, 0',       // Rojo puro
      '178, 34, 34',     // Rojo fuego
      '139, 0, 0',       // Rojo oscuro
      '255, 99, 71'      // Tomate
    ],
    pointerColor: '220, 20, 60'
  },
  {
    name: 'Cyber Teal',
    gradientBackgroundStart: 'rgb(0, 128, 128)',
    gradientBackgroundEnd: 'rgb(0, 139, 139)',
    colors: [
      '0, 255, 255',     // Cyan
      '72, 209, 204',    // Turquesa medio
      '0, 206, 209',     // Turquesa oscuro
      '64, 224, 208',    // Turquesa
      '32, 178, 170',    // Verde mar claro
      '0, 191, 255'      // Azul cielo profundo
    ],
    pointerColor: '0, 255, 255'
  },
  {
    name: 'Golden Thunder',
    gradientBackgroundStart: 'rgb(184, 134, 11)',
    gradientBackgroundEnd: 'rgb(133, 77, 14)',
    colors: [
      '255, 215, 0',     // Dorado
      '255, 140, 0',     // Naranja dorado
      '255, 165, 0',     // Naranja
      '218, 165, 32',    // Vara dorada
      '184, 134, 11',    // Dorado oscuro
      '255, 193, 7'      // Ámbar
    ],
    pointerColor: '255, 215, 0'
  },
  {
    name: 'Violet Vortex',
    gradientBackgroundStart: 'rgb(148, 0, 211)',
    gradientBackgroundEnd: 'rgb(75, 0, 130)',
    colors: [
      '186, 85, 211',    // Orquídea medio
      '138, 43, 226',    // Violeta azul
      '147, 112, 219',   // Púrpura claro
      '123, 104, 238',   // Pizarra azul medio
      '106, 90, 205',    // Pizarra azul
      '72, 61, 139'      // Pizarra azul oscuro
    ],
    pointerColor: '186, 85, 211'
  },
  {
    name: 'Sakura Dreams',
    gradientBackgroundStart: 'rgb(255, 182, 193)',
    gradientBackgroundEnd: 'rgb(255, 105, 180)',
    colors: [
      '255, 182, 193',   // Rosa claro
      '255, 192, 203',   // Rosa
      '255, 105, 180',   // Rosa fuerte
      '255, 20, 147',    // Rosa profundo
      '219, 112, 147',   // Rosa pálido
      '255, 160, 122'    // Salmón claro
    ],
    pointerColor: '255, 105, 180'
  },
  {
    name: 'Arctic Aurora',
    gradientBackgroundStart: 'rgb(0, 191, 255)',
    gradientBackgroundEnd: 'rgb(138, 43, 226)',
    colors: [
      '0, 191, 255',     // Azul cielo profundo
      '30, 144, 255',    // Azul dodger
      '75, 0, 130',      // Índigo
      '138, 43, 226',    // Violeta azul
      '147, 112, 219',   // Púrpura claro
      '176, 196, 222'    // Azul acero claro
    ],
    pointerColor: '0, 191, 255'
  }
];

export const BackgroundGradientAnimation = ({
  gradientBackgroundStart = "rgb(108, 0, 162)",
  gradientBackgroundEnd = "rgb(0, 17, 82)",
  firstColor = "18, 113, 255",
  secondColor = "138, 43, 226",
  thirdColor = "100, 220, 255",
  fourthColor = "200, 50, 50",
  fifthColor = "180, 180, 50",
  pointerColor = "0, 191, 255",
  size = "80%",
  blendingValue = "hard-light",
  children,
  className,
  interactive = true,
  containerClassName
}: BackgroundGradientAnimationProps) => {
  const interactiveRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [curX, setCurX] = useState(0);
  const [currentPaletteIndex, setCurrentPaletteIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentPaletteIndex((currentPaletteIndex + 1) % palettes.length);
    }, 2000); // 2 seconds per palette
    return () => clearInterval(interval);
  }, [currentPaletteIndex]);

  useEffect(() => {
    const currentPalette = palettes[currentPaletteIndex];
    document.body.style.setProperty("--gradient-background-start", currentPalette.gradientBackgroundStart);
    document.body.style.setProperty("--gradient-background-end", currentPalette.gradientBackgroundEnd);
    
    // Mapear correctamente los colores a las variables CSS
    document.body.style.setProperty("--first-color", currentPalette.colors[0]);
    document.body.style.setProperty("--second-color", currentPalette.colors[1]);
    document.body.style.setProperty("--third-color", currentPalette.colors[2]);
    document.body.style.setProperty("--fourth-color", currentPalette.colors[3]);
    document.body.style.setProperty("--fifth-color", currentPalette.colors[4]);
    document.body.style.setProperty("--pointer-color", currentPalette.pointerColor);
  }, [currentPaletteIndex]);
  const [curY, setCurY] = useState(0);
  const [tgX, setTgX] = useState(0);
  const [tgY, setTgY] = useState(0);
  
  // Solo configurar size y blending value (no colores)
  useEffect(() => {
    document.body.style.setProperty("--size", size);
    document.body.style.setProperty("--blending-value", blendingValue);
  }, [size, blendingValue]);

  useEffect(() => {
    if (!interactive) return;

    let animationId: number;
    
    function move() {
      if (!interactiveRef.current) {
        return;
      }
      
      setCurX(prevCurX => {
        const newCurX = prevCurX + (tgX - prevCurX) / 20;
        setCurY(prevCurY => {
          const newCurY = prevCurY + (tgY - prevCurY) / 20;
          
          if (interactiveRef.current) {
            interactiveRef.current.style.transform = `translate(${Math.round(newCurX)}px, ${Math.round(newCurY)}px)`;
          }
          
          return newCurY;
        });
        return newCurX;
      });
      
      animationId = requestAnimationFrame(move);
    }

    animationId = requestAnimationFrame(move);
    
    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [tgX, tgY, interactive]);

  const handleMouseMove = (event: React.MouseEvent<HTMLDivElement>) => {
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      setTgX(event.clientX - rect.left);
      setTgY(event.clientY - rect.top);
    }
  };

  const [isSafari, setIsSafari] = useState(false);
  useEffect(() => {
    setIsSafari(/^((?!chrome|android).)*safari/i.test(navigator.userAgent));
  }, []);

  return (
    <div
      ref={containerRef}
      onMouseMove={handleMouseMove}
      className={cn(
        "h-screen w-screen absolute top-0 left-0 z-0 overflow-hidden max-w-full bg-[linear-gradient(40deg,var(--gradient-background-start),var(--gradient-background-end))]",
        "transition-all duration-1000 ease-in-out",
        containerClassName
      )}>
      <svg className="hidden">
        <defs>
          <filter id="blurMe">
            <feGaussianBlur in="SourceGraphic" stdDeviation="10" result="blur" />
            <feColorMatrix
              in="blur"
              mode="matrix"
              values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 18 -8"
              result="goo" />
            <feBlend in="SourceGraphic" in2="goo" />
          </filter>
        </defs>
      </svg>
      <div className={cn("", className)}>{children}</div>
      <div
        className={cn(
          "gradients-container h-full w-full blur-lg",
          isSafari ? "blur-2xl" : "[filter:url(#blurMe)_blur(40px)]"
        )}>
        <div
          className={cn(
            `absolute [background:radial-gradient(circle_at_center,_var(--first-color)_0,_var(--first-color)_50%)_no-repeat]`,
            `[mix-blend-mode:var(--blending-value)] w-[var(--size)] h-[var(--size)] top-[calc(50%-var(--size)/2)] left-[calc(50%-var(--size)/2)]`,
            `[transform-origin:center_center]`,
            `animate-first`,
            `opacity-100`,
            `transition-all duration-1000 ease-in-out`
          )}></div>
        <div
          className={cn(
            `absolute [background:radial-gradient(circle_at_center,_rgba(var(--second-color),_0.8)_0,_rgba(var(--second-color),_0)_50%)_no-repeat]`,
            `[mix-blend-mode:var(--blending-value)] w-[var(--size)] h-[var(--size)] top-[calc(50%-var(--size)/2)] left-[calc(50%-var(--size)/2)]`,
            `[transform-origin:calc(50%-400px)]`,
            `animate-second`,
            `opacity-100`,
            `transition-all duration-1000 ease-in-out`
          )}></div>
        <div
          className={cn(
            `absolute [background:radial-gradient(circle_at_center,_rgba(var(--third-color),_0.8)_0,_rgba(var(--third-color),_0)_50%)_no-repeat]`,
            `[mix-blend-mode:var(--blending-value)] w-[var(--size)] h-[var(--size)] top-[calc(50%-var(--size)/2)] left-[calc(50%-var(--size)/2)]`,
            `[transform-origin:calc(50%+400px)]`,
            `animate-third`,
            `opacity-100`,
            `transition-all duration-1000 ease-in-out`
          )}></div>
        <div
          className={cn(
            `absolute [background:radial-gradient(circle_at_center,_rgba(var(--fourth-color),_0.8)_0,_rgba(var(--fourth-color),_0)_50%)_no-repeat]`,
            `[mix-blend-mode:var(--blending-value)] w-[var(--size)] h-[var(--size)] top-[calc(50%-var(--size)/2)] left-[calc(50%-var(--size)/2)]`,
            `[transform-origin:calc(50%-200px)]`,
            `animate-fourth`,
            `opacity-70`,
            `transition-all duration-1000 ease-in-out`
          )}></div>
        <div
          className={cn(
            `absolute [background:radial-gradient(circle_at_center,_rgba(var(--fifth-color),_0.8)_0,_rgba(var(--fifth-color),_0)_50%)_no-repeat]`,
            `[mix-blend-mode:var(--blending-value)] w-[var(--size)] h-[var(--size)] top-[calc(50%-var(--size)/2)] left-[calc(50%-var(--size)/2)]`,
            `[transform-origin:calc(50%-800px)_calc(50%+800px)]`,
            `animate-fifth`,
            `opacity-100`,
            `transition-all duration-1000 ease-in-out`
          )}></div>

        {interactive && (
          <div
            ref={interactiveRef}
            className={cn(
              `absolute [background:radial-gradient(circle_at_center,_rgba(var(--pointer-color),_0.8)_0,_rgba(var(--pointer-color),_0)_50%)_no-repeat]`,
              `[mix-blend-mode:var(--blending-value)] w-[var(--size)] h-[var(--size)] -translate-x-1/2 -translate-y-1/2 pointer-events-none`,
              `opacity-70`
            )}></div>
        )}
      </div>
    </div>
  );
};
