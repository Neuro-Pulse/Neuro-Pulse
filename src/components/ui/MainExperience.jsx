import { Loader, OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { Suspense } from "react";
import { Experience } from "./Experience";
import { UI } from "./UI";

function MainExperience() {
  return (
    <>
      <UI />
      <Loader />

      <Canvas
        style={{ width: "100vw", height: "100vh" }}
        shadows
        camera={{ position: [-0.5, 1, 4], fov: 45 }}
      >
        <group position-y={0}>
          <Suspense fallback={null}>
            <Experience />
          </Suspense>
        </group>
      </Canvas>
    </>
  );
}

export default MainExperience;
