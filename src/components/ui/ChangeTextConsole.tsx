"use client";

import { useEffect, useState } from "react";
import { RainbowButton } from "../magicui/rainbow-button";

interface ChangeTextConsoleProps {
  text?: string;
}

export default function ChangeTextConsole({text}: ChangeTextConsoleProps) {
  const [mensaje, setMensaje] = useState("Cargando...");
  const [pumpFunLink, setPumpFunLink] = useState("https://pump.fun/board");

  // Función para generar el link de Pump.fun basado en el CA
  const generatePumpFunLink = (contractAddress: string) => {
    // Si es un CA válido, crear link específico del coin
    if (contractAddress && contractAddress.length > 10 && !contractAddress.includes("Cargando") && !contractAddress.includes("Soon")) {
      return `https://pump.fun/coin/${contractAddress}`;
    }
    // Si no es válido, usar el link base del board
    return "https://pump.fun/board";
  };

  useEffect(() => {
    const fetchMensaje = async () => {
      try {
        const res = await fetch("https://alexey-changetextf-11.deno.dev/text");
        const data = await res.json();
        const contractAddress = data.message || "Contract Adress Soon.";
        
        setMensaje(contractAddress);
        // Generar el link de Pump.fun basado en el CA recibido
        const generatedLink = generatePumpFunLink(contractAddress);
        console.log('🔍 DEBUG - CA recibido:', contractAddress);
        console.log('🔗 DEBUG - Link generado:', generatedLink);
        console.log('✅ DEBUG - CA válido:', contractAddress.length > 10 && !contractAddress.includes("Cargando") && !contractAddress.includes("Soon"));
        setPumpFunLink(generatedLink);
      } catch (err) {
        setMensaje("Contract Adress Soon");
        setPumpFunLink("https://pump.fun/board");
      }
    };

    // Llamamos inmediatamente al cargar
    fetchMensaje();

    // Intervalo cada 5 segundos
    const interval = setInterval(fetchMensaje, 5000);

    return () => clearInterval(interval); // Limpieza
  }, []);

  // Función para manejar el click del botón
  const handleButtonClick = () => {
    // Si el mensaje es un CA válido (más de 10 caracteres), abrir Pump.fun
    if (mensaje && mensaje.length > 10 && !mensaje.includes("Cargando") && !mensaje.includes("Soon")) {
      window.open(pumpFunLink, '_blank', 'noopener,noreferrer');
    }
  };

  return (
    <div className="text-3xl font-bold text-center">
        <RainbowButton
              showCopyIcon={true}
              copyText={mensaje}
              variant="default"
              size="default"
              onClick={handleButtonClick}
              className={mensaje && mensaje.length > 10 && !mensaje.includes("Cargando") && !mensaje.includes("Soon") ? "cursor-pointer hover:scale-105 transition-transform" : ""}
              title={mensaje && mensaje.length > 10 && !mensaje.includes("Cargando") && !mensaje.includes("Soon") ? `Click to view on Pump.fun: ${pumpFunLink}` : "Copy contract address"}
            >
              {mensaje}
            </RainbowButton>
      
    </div>
  );
}
