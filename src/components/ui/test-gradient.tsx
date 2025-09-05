"use client";
import React, { useEffect, useRef, useState } from "react";

export const TestGradient = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const interactiveRef = useRef<HTMLDivElement>(null);
  
  const [mouseX, setMouseX] = useState(0);
  const [mouseY, setMouseY] = useState(0);

  const handleMouseMove = (event: React.MouseEvent<HTMLDivElement>) => {
    console.log("MOUSE MOVE DETECTED!", event.clientX, event.clientY);
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      
      console.log("Setting position:", x, y);
      setMouseX(x);
      setMouseY(y);
      
      if (interactiveRef.current) {
        interactiveRef.current.style.transform = `translate(${x}px, ${y}px)`;
      }
    }
  };

  return (
    <div 
      ref={containerRef}
      onMouseMove={handleMouseMove}
      className="fixed inset-0 bg-purple-900 cursor-crosshair z-10"
      style={{ backgroundColor: "rgb(108, 0, 162)" }}
    >
      <div className="text-white p-4">
        Mouse: {mouseX}, {mouseY}
      </div>
      
      <div
        ref={interactiveRef}
        className="absolute w-32 h-32 bg-red-500 rounded-full opacity-50 pointer-events-none"
        style={{
          transform: `translate(${mouseX}px, ${mouseY}px)`,
          background: "radial-gradient(circle, rgba(255,0,0,0.8) 0%, rgba(255,0,0,0) 100%)"
        }}
      />
    </div>
  );
};
