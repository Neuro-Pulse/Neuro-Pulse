"use client";

import type { AppProps } from "next/app";
import NavBar from "@/header/NavBar";
// import StickyCursor from "@/stickyCursor";
import { useRef } from "react";
import styles from "@/styles/style.module.scss";
import Magnetic from "@/magnetic";
import "@/styles/globals.scss";
import ReactLenis from "@/components/ReactLenis";
import '@/utils/testImageUrls'; // Herramientas de debug globales

import Dock from '../components/dock';
import { VscHome, VscArchive, VscAccount, VscSettingsGear } from 'react-icons/vsc';

function MyApp({ Component, pageProps }: AppProps) {
  const items = [
    { icon: <VscHome size={18} />, label: 'Home', onClick: () => alert('Home!') },
    { icon: <VscArchive size={18} />, label: 'Archive', onClick: () => alert('Archive!') },
    { icon: <VscAccount size={18} />, label: 'Profile', onClick: () => alert('Profile!') },
    { icon: <VscSettingsGear size={18} />, label: 'Settings', onClick: () => alert('Settings!') },
  ];
  const stickyCursorRef = useRef<HTMLDivElement | null>(null); // Referencia correctamente tipadatipada correctamente
  // const stickyCursorRef2 = useRef<HTMLDivElement | null>(null); // Tipar stickyCursorRef2 también si se usa

  return (
    <ReactLenis>
      {/* <NavBar stickyCursorRef={stickyCursorRef} /> */}
     
      {/* Si estás utilizando stickyCursorRef2 en alguna parte, asegúrate de usarlo correctamente */}
      {/* 
      <div className={styles.header}>
        <Magnetic>
          <div ref={stickyCursorRef2} className={styles.bounds}></div>
        </Magnetic>
      </div>
      */}
      {/* <StickyCursor stickyElement={stickyCursorRef} /> */}
      <Component {...pageProps} />
    </ReactLenis>
  );
}

export default MyApp;
