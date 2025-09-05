"use client";
import styles from "./style.module.scss";
import { useState } from "react";
import { motion } from "framer-motion";
import { height } from "../anim";
import Body from "./Body";
import Footer from "./Footer";
import Image from "./Image";
import { usePumpFunLink } from "@/hooks/usePumpFunLink";


export default function Index() {
  const [selectedLink, setSelectedLink] = useState({ isActive: false, index: 0 });
  const { pumpFunLink } = usePumpFunLink();
  
  const links = [
    {
      title: "Home",
      href: "/",
      src: "home.png",
    },
    {
      title: "Pump.fun",
      href: "#",
      src: "home.png",
    },
    {
      title: "Extension",
      href: "#",
      src: "shop.png",
    },
    {
      title: "Whitepaper",
      href: "#",
      src: "home.png",
    },
    {
      title: "Developer",
      href: "#",
      src: "home.png",
    },
    {
      title: "Buy $TECH",
      href: "#",
      src: "contact.png",
    },
    {
      title: "Twitter",
      href: "#",
      src: "contact.png",
    },
    {
      title: "GitHub",
      href: "#",
      src: "lookbook.png",
    },
  ];

  return (
    <motion.div
      variants={height}
      initial="initial"
      animate="enter"
      exit="exit"
      className={styles.nav}
    >
      <div className={styles.wrapper}>
        <div className={styles.container}>
          <Body links={links} selectedLink={selectedLink} setSelectedLink={setSelectedLink} />
          <Footer />
        </div>
        <Image src={links[selectedLink.index].src} isActive={selectedLink.isActive} />
      </div>
    </motion.div>
  );
}
