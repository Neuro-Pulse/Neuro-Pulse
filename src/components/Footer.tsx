import React, { useState, useEffect } from "react"
import { motion } from "framer-motion"

const Footer: React.FC = () => {
  const [showFooter, setShowFooter] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      const scrollTop = window.scrollY;
      const documentHeight = document.documentElement.scrollHeight;
      const windowHeight = window.innerHeight;
      const scrollPercentage = (scrollTop / (documentHeight - windowHeight)) * 100;
      
      // Activar footer cuando el usuario haya scrolleado 93% de la página (casi al final)
      if (scrollPercentage >= 93) {
        setShowFooter(true);
      } else {
        setShowFooter(false);
      }
    };

    window.addEventListener('scroll', handleScroll);
    
    // Verificar al cargar la página
    handleScroll();
    
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  if (!showFooter) {
    return null;
  }

  return (
    <motion.div 
      initial={{ opacity: 0, y: 100 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 100 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className="sticky bottom-0 left-0 w-full h-80 bg-[#EBEFEF] flex justify-center items-center z-0" 
    >
      <div className="relative overflow-hidden w-full h-full flex justify-end px-12 text-right items-start py-12 text-black">
        <div className="flex flex-row space-x-12 sm:space-x-16 md:space-x-24 text-sm sm:text-lg md:text-xl">
          <ul className="space-y-2">
            <li className="hover:text-gray-600 hover:underline cursor-pointer transition-all duration-300 ease-out transform hover:translate-x-1">
              Home
            </li>
            <li className="hover:text-gray-600 hover:underline cursor-pointer transition-all duration-300 ease-out transform hover:translate-x-1">
              Docs
            </li>
            <li className="hover:text-gray-600 hover:underline cursor-pointer transition-all duration-300 ease-out transform hover:translate-x-1">
              Comps
            </li>
            <li className="hover:text-gray-600 hover:underline cursor-pointer transition-all duration-300 ease-out transform hover:translate-x-1">
              Whitepaper
            </li>
          </ul>
          <ul className="space-y-2">
            <li className="hover:text-gray-600 hover:underline cursor-pointer transition-all duration-300 ease-out transform hover:translate-x-1">
              Github
            </li>
            <li className="hover:text-gray-600 hover:underline cursor-pointer transition-all duration-300 ease-out transform hover:translate-x-1">
              Instagram
            </li>
            <li className="hover:text-gray-600 hover:underline cursor-pointer transition-all duration-300 ease-out transform hover:translate-x-1">
              X (Twitter)
            </li>
          </ul>
        </div>
        <h2 className="absolute bottom-0 left-0 translate-y-1/3 sm:text-[192px] text-[80px] ">
          ❤
        </h2>
      </div>
    </motion.div>
  )
}

export default Footer
