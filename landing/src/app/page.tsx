import Link from "next/link";
import { ArrowRight, Layers, Code, Download, Zap } from "lucide-react";

export default function LandingPage() {
  return (
    <div className="flex min-h-screen flex-col">
      {/* Test Tailwind */}

      {/* Hero Section */}
      <section className="bg-gradient-to-b from-purple-50 to-white py-20 px-4">
        <div className="container mx-auto max-w-5xl">
          <div className="flex flex-col items-center text-center space-y-8">
            <h1 className="text-5xl md:text-6xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-600 to-purple-400">
              NeuroBlock
            </h1>
            <p className="text-xl md:text-2xl text-gray-700 max-w-2xl">
              Your Visual AI Model Builder
            </p>
            <div className="flex flex-col sm:flex-row gap-4 mt-8">
              <Link
                href="https://app.neuroblock.co/newbuild"
                className="px-8 py-3 rounded-lg bg-purple-600 text-white font-medium hover:bg-purple-700 transition-colors"
              >
                Try Now
              </Link>
              <Link
                href="https://app.neuroblock.co/tutorial"
                className="px-8 py-3 rounded-lg border border-purple-200 text-purple-600 font-medium hover:bg-purple-50 transition-colors"
              >
                Watch Tutorial
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-4">
        <div className="container mx-auto max-w-5xl">
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-16 text-gray-800">
            Everything you need to build AI models
          </h2>

          <div className="grid md:grid-cols-2 gap-12">
            <div className="bg-white p-8 rounded-xl shadow-sm border border-purple-100 hover:shadow-md transition-shadow">
              <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mb-6">
                <Layers className="text-purple-600 w-6 h-6" />
              </div>
              <h3 className="text-xl font-semibold mb-3 text-gray-800">
                Drag-and-Drop Interface
              </h3>
              <p className="text-gray-600">
                Build your AI models visually with our intuitive drag-and-drop
                interface. No coding required.
              </p>
            </div>

            <div className="bg-white p-8 rounded-xl shadow-sm border border-purple-100 hover:shadow-md transition-shadow">
              <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mb-6">
                <Zap className="text-purple-600 w-6 h-6" />
              </div>
              <h3 className="text-xl font-semibold mb-3 text-gray-800">
                Prebuilt Templates
              </h3>
              <p className="text-gray-600">
                Get started quickly with our collection of pre-built templates
                for common AI model architectures.
              </p>
            </div>

            <div className="bg-white p-8 rounded-xl shadow-sm border border-purple-100 hover:shadow-md transition-shadow">
              <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mb-6">
                <Download className="text-purple-600 w-6 h-6" />
              </div>
              <h3 className="text-xl font-semibold mb-3 text-gray-800">
                Export in Multiple Formats
              </h3>
              <p className="text-gray-600">
                Export your models in various formats for deployment across
                different platforms and frameworks.
              </p>
            </div>

            <div className="bg-white p-8 rounded-xl shadow-sm border border-purple-100 hover:shadow-md transition-shadow">
              <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mb-6">
                <Code className="text-purple-600 w-6 h-6" />
              </div>
              <h3 className="text-xl font-semibold mb-3 text-gray-800">
                Advanced Customization
              </h3>
              <p className="text-gray-600">
                Fine-tune your models with advanced parameters while maintaining
                the simplicity of visual design.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="bg-purple-50 py-16 px-4">
        <div className="container mx-auto max-w-5xl text-center">
          <h2 className="text-2xl md:text-3xl font-bold mb-6 text-gray-800">
            Ready to build your AI model?
          </h2>
          <Link
            href="https://app.neuroblock.co/newbuild"
            className="inline-flex items-center px-8 py-3 rounded-lg bg-purple-600 text-white font-medium hover:bg-purple-700 transition-colors"
          >
            Get Started <ArrowRight className="ml-2 h-4 w-4" />
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-4 border-t border-gray-100">
        <div className="container mx-auto max-w-5xl">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="text-purple-600 font-semibold text-lg mb-4 md:mb-0">
              NeuroBlock
            </div>
            <div className="flex space-x-6">
              <Link
                href="#"
                className="text-gray-600 hover:text-purple-600 transition-colors"
              >
                About
              </Link>
              <Link
                href="#"
                className="text-gray-600 hover:text-purple-600 transition-colors"
              >
                Documentation
              </Link>
              <Link
                href="#"
                className="text-gray-600 hover:text-purple-600 transition-colors"
              >
                Contact
              </Link>
            </div>
          </div>
          <div className="mt-8 text-center text-gray-500 text-sm">
            © {new Date().getFullYear()} NeuroBlock. All rights reserved.
          </div>
        </div>
      </footer>
    </div>
  );
}
