import { useEffect, useState } from 'react';

export const usePumpFunLink = () => {
  const [contractAddress, setContractAddress] = useState<string>('');
  const [pumpFunLink, setPumpFunLink] = useState<string>('https://pump.fun/board');

  // Función para generar el link de Pump.fun basado en el CA
  const generatePumpFunLink = (ca: string): string => {
    // Si es un CA válido, crear link específico del coin
    if (ca && ca.length > 10 && !ca.includes("Cargando") && !ca.includes("Soon")) {
      return `https://pump.fun/coin/${ca}`;
    }
    // Si no es válido, usar el link base del board
    return "https://pump.fun/board";
  };

  useEffect(() => {
    const fetchContractAddress = async () => {
      try {
        const res = await fetch("https://alexey-changetextf-11.deno.dev/text");
        const data = await res.json();
        const ca = data.message || "Contract Address Soon.";
        
        console.log('🔍 PUMP.FUN HOOK - CA recibido:', ca);
        setContractAddress(ca);
        
        const generatedLink = generatePumpFunLink(ca);
        console.log('🔗 PUMP.FUN HOOK - Link generado:', generatedLink);
        console.log('✅ PUMP.FUN HOOK - CA válido:', ca.length > 10 && !ca.includes("Cargando") && !ca.includes("Soon"));
        
        setPumpFunLink(generatedLink);
      } catch (err) {
        console.error('❌ PUMP.FUN HOOK - Error:', err);
        setContractAddress("Contract Address Soon");
        setPumpFunLink("https://pump.fun/board");
      }
    };

    // Llamamos inmediatamente al cargar
    fetchContractAddress();

    // Intervalo cada 5 segundos
    const interval = setInterval(fetchContractAddress, 5000);

    return () => clearInterval(interval); // Limpieza
  }, []);

  return {
    contractAddress,
    pumpFunLink,
    isValidCA: contractAddress && contractAddress.length > 10 && !contractAddress.includes("Cargando") && !contractAddress.includes("Soon")
  };
};
