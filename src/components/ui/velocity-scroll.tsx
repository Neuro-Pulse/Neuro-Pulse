import React from "react";
import { wrap } from "framer-motion";
import {
  motion,
  useScroll,
  useSpring,
  useTransform,
  useMotionValue,
  useVelocity,
  useAnimationFrame,
} from "framer-motion";

interface ParallaxProps {
  children: string;
  baseVelocity: number;
}

// Componente de texto marquee con efecto de velocidad
function ParallaxText({ children, baseVelocity = 100 }: ParallaxProps) {
  const baseX = useMotionValue(0);
  const { scrollY } = useScroll();
  const scrollVelocity = useVelocity(scrollY);
  const smoothVelocity = useSpring(scrollVelocity, {
    damping: 30,
    stiffness: 200,
  });
  const velocityFactor = useTransform(smoothVelocity, [0, 1000], [0, 3.1185], {
    clamp: false,
  });

  const x = useTransform(baseX, (v) => `${wrap(-25, 0, v)}%`);

  const directionFactor = React.useRef<number>(1);
  useAnimationFrame((t, delta) => {
    let moveBy = directionFactor.current * baseVelocity * (delta / 1000);

    if (velocityFactor.get() < 0) {
      directionFactor.current = -1;
    } else if (velocityFactor.get() > 0) {
      directionFactor.current = 1;
    }

    moveBy += directionFactor.current * moveBy * velocityFactor.get();

    baseX.set(baseX.get() + moveBy);
  });

  return (
    <div className="parallax w-full overflow-hidden relative flex items-center justify-center">
      <motion.div
        className="scroller flex whitespace-nowrap font-display text-8xl font-bold tracking-[-0.02em] drop-shadow-sm md:text-[12rem] md:leading-[10rem] will-change-transform"
        style={{ 
          x,
          transformStyle: "preserve-3d",
          backfaceVisibility: "hidden",
          perspective: 1000
        }}
      >
        {Array(12).fill(null).map((_, i) => (
          <span key={i} className="mr-8 inline-block">{children}</span>
        ))}
      </motion.div>
    </div>
  );
}

// Componente principal - mismo marquee text por 100vh normales + 70vh sticky
export function VelocityScroll() {
  return (
    <div className="w-full">
      {/* Contenedor total para sticky effect: 100vh + 70vh = 170vh */}
      <div className="relative h-[170vh]">
        {/* El marquee text se ve por 100vh normales, luego 70vh sticky */}
        <div className="sticky top-0 h-screen flex flex-col items-center justify-center font-formula-condensed uppercase text-black bg-white">
          <ParallaxText baseVelocity={2.2}>Neuro Pulse</ParallaxText>
          <ParallaxText baseVelocity={-2.2}>Neuro Pulse</ParallaxText>
        </div>
      </div>
    </div>
  );
}
