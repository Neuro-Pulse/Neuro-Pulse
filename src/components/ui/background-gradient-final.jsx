import React from "react";
import { BackgroundGradientAnimation } from "../ui/background-gradient-animation";
 
export function BackgroundGradientAnimationDemo() {
  return (
    <div className="overflow-x-hidden">
      <BackgroundGradientAnimation>
        <div className="absolute z-50 inset-0 flex items-center justify-center text-white font-bold px-4 pointer-events-none text-3xl text-center md:text-4xl lg:text-7xl">
         
        </div>
      </BackgroundGradientAnimation>
    </div>
  );
}
