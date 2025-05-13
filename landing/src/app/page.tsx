
import {
  ArrowRightIcon,
  SparklesIcon,
  CubeIcon,
  ArrowDownTrayIcon,
} from "@heroicons/react/24/outline";
import "./styles/landing.css";

export default function Home() {
  return (
    <div className="min-h-screen gradient-bg">
      {/* Hero Section */}
      <div className="relative isolate">
        {/* Background effects */}
        <div className="absolute inset-x-0 -top-40 -z-10 transform-gpu overflow-hidden blur-3xl sm:-top-80">
          <div className="relative left-[calc(50%-11rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 rotate-[30deg] bg-gradient-to-tr from-indigo-200 to-purple-200 opacity-30 sm:left-[calc(50%-30rem)] sm:w-[72.1875rem]" />
        </div>

        <div className="mx-auto max-w-7xl px-6 py-24 sm:py-32 lg:px-8">
          <div className="mx-auto max-w-2xl text-center">
            <h1 className="text-4xl font-bold tracking-tight sm:text-6xl gradient-text animate-float">
              NeuroBlock
            </h1>
            <p className="mt-6 text-lg leading-8 text-gray-600">
              Your Visual AI Model Builder
            </p>
            <div className="mt-10 flex items-center justify-center gap-x-6">
              <a
                href="http://localhost:5173"
                className="gradient-button group relative inline-flex items-center justify-center px-8 py-3 font-medium tracking-wide text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              >
                Try Now
                <span className="shine-effect absolute inset-0 rounded-lg" />
              </a>
              <a
                href="http://localhost:5173/tutorial"
                className="group inline-flex items-center text-sm font-semibold leading-6 text-gray-900 hover:text-indigo-600 transition-colors duration-200"
              >
                Tutorial
                <ArrowRightIcon className="ml-2 h-4 w-4 transition-transform duration-200 group-hover:translate-x-1" />
              </a>
            </div>
          </div>
        </div>

        {/* Background effects */}
        <div className="absolute inset-x-0 top-[calc(100%-13rem)] -z-10 transform-gpu overflow-hidden blur-3xl sm:top-[calc(100%-30rem)]">
          <div className="relative left-[calc(50%+3rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 bg-gradient-to-tr from-indigo-200 to-purple-200 opacity-30 sm:left-[calc(50%+36rem)] sm:w-[72.1875rem]" />
        </div>
      </div>

      {/* Features Section */}
      <div className="py-24 sm:py-32">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="mx-auto max-w-2xl lg:text-center">
            <h2 className="text-base font-semibold leading-7 text-indigo-600">
              Features
            </h2>
            <p className="mt-2 text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
              Everything you need to build AI models
            </p>
          </div>
          <div className="mx-auto mt-16 max-w-2xl sm:mt-20 lg:mt-24 lg:max-w-none">
            <dl className="grid max-w-xl grid-cols-1 gap-x-8 gap-y-16 lg:max-w-none lg:grid-cols-3">
              {/* Feature 1 */}
              <div className="feature-card glass-card group relative flex flex-col items-start rounded-2xl p-6 hover-glow">
                <div className="mb-4 rounded-lg bg-indigo-600 p-2 ring-1 ring-indigo-600/10">
                  <SparklesIcon className="feature-icon h-6 w-6 text-white" />
                </div>
                <dt className="text-lg font-semibold leading-7 text-gray-900">
                  Drag-and-Drop Interface
                </dt>
                <dd className="mt-4 flex flex-auto flex-col text-base leading-7 text-gray-600">
                  <p className="flex-auto">
                    Build your AI models visually with our intuitive
                    drag-and-drop interface. No coding required.
                  </p>
                </dd>
              </div>

              {/* Feature 2 */}
              <div className="feature-card glass-card group relative flex flex-col items-start rounded-2xl p-6 hover-glow">
                <div className="mb-4 rounded-lg bg-indigo-600 p-2 ring-1 ring-indigo-600/10">
                  <CubeIcon className="feature-icon h-6 w-6 text-white" />
                </div>
                <dt className="text-lg font-semibold leading-7 text-gray-900">
                  Prebuilt Templates
                </dt>
                <dd className="mt-4 flex flex-auto flex-col text-base leading-7 text-gray-600">
                  <p className="flex-auto">
                    Get started quickly with our collection of pre-built
                    templates for common AI model architectures.
                  </p>
                </dd>
              </div>

              {/* Feature 3 */}
              <div className="feature-card glass-card group relative flex flex-col items-start rounded-2xl p-6 hover-glow">
                <div className="mb-4 rounded-lg bg-indigo-600 p-2 ring-1 ring-indigo-600/10">
                  <ArrowDownTrayIcon className="feature-icon h-6 w-6 text-white" />
                </div>
                <dt className="text-lg font-semibold leading-7 text-gray-900">
                  Export in Multiple Formats
                </dt>
                <dd className="mt-4 flex flex-auto flex-col text-base leading-7 text-gray-600">
                  <p className="flex-auto">
                    Export your models in various formats for deployment across
                    different platforms and frameworks.
                  </p>
                </dd>
              </div>
            </dl>
          </div>
        </div>
      </div>
    </div>
  );
}
