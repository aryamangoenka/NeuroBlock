import Link from "next/link";
import { ArrowRightIcon } from "@heroicons/react/24/outline";

export default function HomePage() {
  return (
    <main className="relative min-h-screen overflow-x-hidden">
      <div className="absolute inset-0 z-0 hexagon-grid"></div>

      {/* Hero */}
      <section className="relative z-10 flex flex-col items-center justify-center text-center px-4 pt-32 pb-24">
        <h1 className="text-5xl md:text-7xl font-bold leading-tight neon-text-glow">
          Build AI Visually with <span className="text-[#39ff14]">NeuroBlock</span>
        </h1>
        <p className="mt-6 max-w-2xl text-lg text-gray-300">
          Drag-and-drop layers. Train live. Export anywhere.
        </p>
        <div className="mt-8 flex flex-wrap justify-center gap-4">
          <a href="https://app.neuroblock.co/newbuild" className="primary-button btn-pulse">
            Launch Builder
          </a>
          <a href="https://app.neuroblock.co/tutorial" className="secondary-button">
            Watch Tutorial
          </a>
        </div>
      </section>

      {/* Neural Visualization */}
      <section className="relative z-10 py-16">
        <div className="container mx-auto px-4">
          <div className="aspect-[16/9] bg-black/40 rounded-xl overflow-hidden relative border border-[#39ff14]/30">
            <div className="neural-visualization w-full h-full">
              <div className="node input-node" style={{ left: "10%", top: "50%" }}></div>
              <div className="node hidden-node-1" style={{ left: "35%", top: "30%" }}></div>
              <div className="node hidden-node-2" style={{ left: "35%", top: "70%" }}></div>
              <div className="node output-node" style={{ left: "75%", top: "50%" }}></div>

              <div className="connection connection-1" style={{ left: "14%", top: "50%", width: "21%" }}></div>
              <div className="connection connection-2" style={{ left: "14%", top: "50%", width: "21%", transform: "rotate(-25deg)" }}></div>
              <div className="connection connection-3" style={{ left: "14%", top: "50%", width: "21%", transform: "rotate(25deg)" }}></div>
              <div className="connection connection-4" style={{ left: "39%", top: "30%", width: "28%" }}></div>
              <div className="connection connection-5" style={{ left: "39%", top: "70%", width: "28%" }}></div>
            </div>
          </div>
        </div>
      </section>

      {/* Terminal Simulation */}
      <section className="relative z-10 py-16">
        <div className="container mx-auto px-4">
          <div className="terminal max-w-3xl mx-auto">
            <div className="terminal-line">
              <span className="terminal-prompt">$</span> neuroblock init
            </div>
            <div className="terminal-line text-[#39ff14]">
              ✓ Initialized successfully
            </div>
            <div className="terminal-line">
              <span className="terminal-prompt">$</span> neuroblock launch --visual
            </div>
            <div className="terminal-line text-[#39ff14]">
              ✓ Visual Builder Ready
            </div>
            <div className="terminal-line">
              <span className="terminal-prompt">$</span> start-training
            </div>
            <div className="terminal-line">
              Training...
              <span className="terminal-cursor"></span>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="relative z-10 text-center py-24">
        <h2 className="text-4xl md:text-5xl font-bold mb-6 neon-text-glow">
          Ready to launch your model?
        </h2>
        <p className="text-lg text-gray-400 max-w-xl mx-auto mb-8">
          Visualize. Build. Train. Deploy. All in one place.
        </p>
        <a href="https://app.neuroblock.co/newbuild" className="primary-button btn-pulse">
          Get Started
        </a>
      </section>
    </main>
  );
}
