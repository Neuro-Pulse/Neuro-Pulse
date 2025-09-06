import { motion } from "framer-motion";

export const HeroSection = () => {
  return (
    <div className="min-h-screen flex items-center justify-center px-4 sm:px-6 md:px-8">
      <div className="text-center max-w-7xl mx-auto pb-12 sm:pb-16 md:pb-20 lg:pb-24">
        {/* Main Hero Text - Responsive Layout */}
        <div className="flex flex-col lg:flex-row items-center justify-center gap-4 sm:gap-6 md:gap-8 lg:gap-16">
          {/* OUTSMART */}
          <span className="text-[4rem] sm:text-[6rem] md:text-[8rem] lg:text-[10rem] xl:text-[12rem] 2xl:text-[14rem] font-formula-bold leading-[0.9] lg:leading-[1] uppercase order-1 lg:order-1">
            OUTSMART
          </span>
          
          {/* Center Text Block */}
          <div className="flex flex-col items-center text-white text-sm sm:text-base md:text-lg lg:text-xl leading-[1] uppercase mb-4 lg:mb-[20px] bb-manual-mono-force order-3 lg:order-2">
            <div className="mb-0 text-center">Neuro Pulse TRACKER ($NEURO)</div>
            <div className="flex items-center justify-center whitespace-nowrap">
              <span className="translate-x-0 sm:translate-x-[-20px] lg:translate-x-[-50px]">+</span>
              <span className="mx-2 text-center">IS POWERING THE FUTURE</span>
              <span className="translate-x-0 sm:translate-x-[20px] lg:translate-x-[50px]">+</span>
            </div>
            <div className="text-center">OF SMART TRADING</div>
          </div>
          
          {/* THE */}
          <span className="text-[4rem] sm:text-[6rem] md:text-[8rem] lg:text-[10rem] xl:text-[12rem] 2xl:text-[14rem] font-formula-bold leading-[0.9] lg:leading-[0.7] uppercase order-2 lg:order-3">
            THE
          </span>
        </div>
        
        {/* ALPHA WALLETS */}
        <div className="text-[4rem] sm:text-[6rem] md:text-[8rem] lg:text-[10rem] xl:text-[12rem] 2xl:text-[14rem] font-formula-bold leading-[0.9] lg:leading-[0.7] uppercase mt-2 sm:mt-4 lg:mt-2 pb-6 sm:pb-8 lg:pb-10">
          ALPHA WALLETS
        </div>
        
        {/* Buy ARX Button - DESACTIVADO */}
        {/* <div className="flex justify-center items-center mt-6 sm:mt-8 lg:mt-10 pointer-events-auto">
          <motion.a 
            href="https://dexscreener.com/"
            target="_blank"
            rel="noopener noreferrer"
            className="group bg-black rounded-[15px] px-4 sm:px-6 lg:px-8 py-2 sm:py-3 lg:py-2 flex items-center justify-center gap-2 sm:gap-3 lg:gap-4 relative overflow-hidden cursor-pointer"
            whileHover={{ 
              scale: 1.05,
              backgroundColor: "#1a1a1a",
              boxShadow: "0 10px 30px rgba(0, 0, 0, 0.3)"
            }}
            whileTap={{ scale: 0.95 }}
            transition={{ 
              type: "spring", 
              stiffness: 300, 
              damping: 20 
            }}
          >
            {/* Animated background glow */}
            {/* <motion.div
              className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent"
              initial={{ x: "-100%" }}
              whileHover={{ x: "100%" }}
              transition={{ duration: 0.6, ease: "easeInOut" }}
            /> */}
            
            {/* <motion.p 
              className="text-white group-hover:text-[#CDFE00] text-sm sm:text-base lg:text-xl font-bb-mono uppercase tracking-wider relative z-10"
            >
              DEXSCREENER
            </motion.p> */}
            
            {/* <motion.img 
              src="dexscreener.png"
              alt="Dexscreener"
              className="w-8 h-8 sm:w-9 sm:h-9 lg:w-10 lg:h-10 rounded-full object-contain relative z-10"
              whileHover={{ 
                scale: 1.1
              }}
              transition={{ 
                scale: { duration: 0.2 }
              }}
            /> */}
          {/* </motion.a>
        </div> */}
      </div>
    </div>
  );
};
