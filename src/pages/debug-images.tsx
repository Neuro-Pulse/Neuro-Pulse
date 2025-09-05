import { NextPage } from 'next';
import ImageDebugger from '@/components/ui/ImageDebugger';

const DebugImagesPage: NextPage = () => {
  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4">
        <h1 className="text-3xl font-bold text-center mb-8">🔧 Image Debug Tool</h1>
        <ImageDebugger />
        
        <div className="mt-8 text-center text-sm text-gray-600">
          <p>
            Esta herramienta te permite probar URLs de imágenes problemáticas 
            y ver los fallbacks automáticos en acción.
          </p>
          <p className="mt-2">
            <strong>Abre la consola del navegador</strong> para ver los logs detallados.
          </p>
        </div>
      </div>
    </div>
  );
};

export default DebugImagesPage;
