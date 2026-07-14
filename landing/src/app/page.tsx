import Image from "next/image";

const APP_URL = process.env.NEXT_PUBLIC_APP_URL;
const REPO_URL = "https://github.com/aryamangoenka/NeuroBlock";
const primaryHref = APP_URL ?? REPO_URL;
const primaryLabel = APP_URL ? "Open the builder" : "Get the code";

const LAYERS: { name: string; hue: string; learns: boolean }[] = [
  { name: "Input", hue: "var(--n-input)", learns: false },
  { name: "Conv2D", hue: "var(--n-conv)", learns: true },
  { name: "MaxPooling", hue: "var(--n-pool)", learns: false },
  { name: "Flatten", hue: "var(--n-flat)", learns: false },
  { name: "Dense", hue: "var(--n-dense)", learns: true },
  { name: "Dropout", hue: "var(--n-drop)", learns: false },
  { name: "Output", hue: "var(--n-out)", learns: true },
];

const STEPS = [
  { n: "01", title: "Pick a dataset", body: "MNIST, CIFAR-10, Iris — or upload your own CSV or images." },
  { n: "02", title: "Connect blocks", body: "Drag layers onto the canvas and wire input to output." },
  { n: "03", title: "Watch it learn", body: "Loss and accuracy stream in live, epoch by epoch." },
  { n: "04", title: "Export real code", body: "Python, notebook, Keras, or PyTorch — the network you built." },
];

function Rail({ hue, learns }: { hue: string; learns: boolean }) {
  return (
    <span
      className="inline-block h-5 w-[5px] rounded-[2px]"
      style={learns ? { background: hue } : { border: `1.5px solid ${hue}` }}
    />
  );
}

export default function LandingPage() {
  return (
    <div className="min-h-screen">
      {/* nav */}
      <header className="border-b" style={{ borderColor: "var(--line)", background: "var(--card)" }}>
        <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-5">
          <span className="f-display text-lg font-bold">
            NeuroBlock<span style={{ color: "var(--citrus)" }}>.</span>
          </span>
          <nav className="flex items-center gap-2">
            <a
              href={REPO_URL}
              className="rounded-lg px-3.5 py-2 text-[13.5px] font-medium transition-colors hover:bg-[var(--raised)]"
              style={{ color: "var(--muted)" }}
            >
              GitHub
            </a>
            <a
              href={primaryHref}
              className="rounded-lg px-4 py-2 text-[13.5px] font-semibold text-white transition-colors"
              style={{ background: "var(--citrus)" }}
            >
              {primaryLabel}
            </a>
          </nav>
        </div>
      </header>

      {/* hero */}
      <section className="dotgrid border-b" style={{ borderColor: "var(--line)" }}>
        <div className="mx-auto max-w-6xl px-5 pb-16 pt-16 md:pt-24">
          <p className="f-mono mb-5 text-[11px] uppercase tracking-[0.14em]" style={{ color: "var(--citrus)" }}>
            Neural networks, by hand
          </p>
          <h1 className="f-display max-w-3xl text-4xl font-bold leading-[1.05] md:text-6xl">
            Build a network.
            <br />
            Watch it learn.
          </h1>
          <p className="mt-6 max-w-xl text-lg leading-relaxed" style={{ color: "var(--muted)" }}>
            Design, train, and export real neural networks by connecting blocks
            on a canvas. No code required — but real code comes out.
          </p>
          <div className="mt-8 flex flex-wrap items-center gap-3">
            <a
              href={primaryHref}
              className="rounded-[9px] px-6 py-3 text-[15px] font-semibold text-white transition-colors"
              style={{ background: "var(--citrus)" }}
            >
              {primaryLabel}
            </a>
            <a
              href={`${REPO_URL}#quick-start`}
              className="card-shadow rounded-[9px] border bg-white px-6 py-3 text-[15px] font-medium transition-colors hover:bg-[var(--raised)]"
              style={{ borderColor: "var(--line-strong)" }}
            >
              Run it yourself
            </a>
            <span className="f-mono ml-1 text-xs" style={{ color: "var(--muted)" }}>
              open source · MIT
            </span>
          </div>

          {/* product shot */}
          <div
            className="card-shadow mt-14 overflow-hidden rounded-xl border bg-white"
            style={{ borderColor: "var(--line)" }}
          >
            <div
              className="flex h-9 items-center gap-1.5 border-b px-4"
              style={{ borderColor: "var(--line)", background: "var(--card)" }}
            >
              <span className="h-2.5 w-2.5 rounded-full" style={{ background: "var(--line-strong)" }} />
              <span className="h-2.5 w-2.5 rounded-full" style={{ background: "var(--line-strong)" }} />
              <span className="h-2.5 w-2.5 rounded-full" style={{ background: "var(--line-strong)" }} />
              <span className="f-mono ml-3 text-[11px]" style={{ color: "var(--muted)" }}>
                neuroblock — mnist-classifier-01
              </span>
            </div>
            <Image
              src="/builder.png"
              alt="The NeuroBlock builder: layer palette, canvas with connected nodes, and live hyperparameters"
              width={2000}
              height={1143}
              priority
              className="block w-full"
            />
          </div>
        </div>
      </section>

      {/* the color rule — the product's signature idea */}
      <section className="border-b" style={{ borderColor: "var(--line)", background: "var(--card)" }}>
        <div className="mx-auto grid max-w-6xl gap-10 px-5 py-16 md:grid-cols-2 md:items-center">
          <div>
            <h2 className="f-display text-2xl font-semibold md:text-3xl">Color that teaches</h2>
            <p className="mt-4 leading-relaxed" style={{ color: "var(--muted)" }}>
              Every block carries a rule you can see: a <strong style={{ color: "var(--ink)" }}>filled rail</strong> means
              the layer learns — it has trainable parameters. A{" "}
              <strong style={{ color: "var(--ink)" }}>hollow rail</strong> only transforms its input. Count the filled
              blocks and you know where your parameters live. The palette is
              colorblind-validated, so the rule holds for everyone in the room.
            </p>
          </div>
          <div className="flex flex-wrap gap-2.5">
            {LAYERS.map((l) => (
              <span
                key={l.name}
                className="card-shadow inline-flex items-center gap-2.5 rounded-[10px] border bg-white px-3.5 py-2.5 text-[13.5px] font-medium"
                style={{ borderColor: "var(--line)" }}
              >
                <Rail hue={l.hue} learns={l.learns} />
                {l.name}
                {l.learns && (
                  <span className="f-mono text-[9px] uppercase tracking-wider" style={{ color: l.hue }}>
                    learns
                  </span>
                )}
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* how it works */}
      <section className="border-b" style={{ borderColor: "var(--line)" }}>
        <div className="mx-auto max-w-6xl px-5 py-16">
          <h2 className="f-display text-2xl font-semibold md:text-3xl">Four steps, no setup</h2>
          <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {STEPS.map((s) => (
              <div
                key={s.n}
                className="card-shadow rounded-xl border bg-white p-5"
                style={{ borderColor: "var(--line)" }}
              >
                <span className="f-mono text-[11px]" style={{ color: "var(--citrus)" }}>
                  {s.n}
                </span>
                <h3 className="f-display mt-2 text-[15px] font-semibold">{s.title}</h3>
                <p className="mt-1.5 text-[13px] leading-relaxed" style={{ color: "var(--muted)" }}>
                  {s.body}
                </p>
              </div>
            ))}
          </div>

          {/* live metrics strip */}
          <div
            className="card-shadow f-mono mt-10 flex flex-wrap items-center gap-x-7 gap-y-2 overflow-x-auto rounded-xl border bg-white px-5 py-3.5 text-[12.5px]"
            style={{ borderColor: "var(--line)", color: "var(--muted)" }}
          >
            <span className="flex items-center gap-2" style={{ color: "var(--citrus)" }}>
              <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full" style={{ background: "var(--citrus)" }} />
              training
            </span>
            <span>
              epoch <b style={{ color: "var(--ink)" }}>07/10</b>
            </span>
            <span>
              loss <b style={{ color: "var(--ink)" }}>0.1284</b>
            </span>
            <span>
              val_acc <b style={{ color: "var(--ok)" }}>0.9612</b>
            </span>
            <span>
              params <b style={{ color: "var(--ink)" }}>101,770</b>
            </span>
            <span className="hidden md:inline">
              out <b style={{ color: "var(--ink)" }}>(None, 26, 26, 32)</b>
            </span>
          </div>
        </div>
      </section>

      {/* classroom + export */}
      <section className="border-b" style={{ borderColor: "var(--line)", background: "var(--card)" }}>
        <div className="mx-auto grid max-w-6xl gap-12 px-5 py-16 md:grid-cols-2">
          <div>
            <h2 className="f-display text-2xl font-semibold">Made for classrooms</h2>
            <p className="mt-4 leading-relaxed" style={{ color: "var(--muted)" }}>
              NeuroBlock runs the intro-to-deep-learning sessions of the Turing
              summer program at UMass Amherst CICS. Thirty students share one
              server: uploads are isolated per session, nothing to install, and
              the whole class watches their networks train at the same time.
            </p>
          </div>
          <div>
            <h2 className="f-display text-2xl font-semibold">The exit ramp is real code</h2>
            <p className="mt-4 leading-relaxed" style={{ color: "var(--muted)" }}>
              When a model works, students take it with them — a runnable Python
              script, a Jupyter notebook, a Keras file, or a PyTorch
              implementation of the exact network they drew. The canvas is the
              on-ramp; the export is the point.
            </p>
            <div className="f-mono mt-5 flex flex-wrap gap-2 text-[12px]">
              {[".py", ".ipynb", ".keras", "savedmodel", "pytorch"].map((f) => (
                <span
                  key={f}
                  className="rounded-md border px-2.5 py-1"
                  style={{ borderColor: "var(--line)", background: "var(--raised)", color: "var(--ink)" }}
                >
                  {f}
                </span>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="dotgrid">
        <div className="mx-auto flex max-w-6xl flex-col items-start justify-between gap-6 px-5 py-16 md:flex-row md:items-center">
          <div>
            <h2 className="f-display text-2xl font-semibold md:text-3xl">Ready when your class is.</h2>
            <p className="mt-2" style={{ color: "var(--muted)" }}>
              One Docker container. One URL. The deployment guide takes fifteen minutes.
            </p>
          </div>
          <div className="flex gap-3">
            <a
              href={primaryHref}
              className="rounded-[9px] px-6 py-3 text-[15px] font-semibold text-white"
              style={{ background: "var(--citrus)" }}
            >
              {primaryLabel}
            </a>
            <a
              href={`${REPO_URL}/blob/main/docs/DEPLOYMENT.md`}
              className="card-shadow rounded-[9px] border bg-white px-6 py-3 text-[15px] font-medium"
              style={{ borderColor: "var(--line-strong)" }}
            >
              Deployment guide
            </a>
          </div>
        </div>
      </section>

      {/* footer */}
      <footer className="border-t" style={{ borderColor: "var(--line)", background: "var(--card)" }}>
        <div
          className="mx-auto flex max-w-6xl flex-col items-start justify-between gap-3 px-5 py-8 text-[13px] md:flex-row md:items-center"
          style={{ color: "var(--muted)" }}
        >
          <span className="f-display text-[15px] font-bold" style={{ color: "var(--ink)" }}>
            NeuroBlock<span style={{ color: "var(--citrus)" }}>.</span>
          </span>
          <div className="flex gap-6">
            <a href={REPO_URL} className="hover:underline">
              GitHub
            </a>
            <a href={`${REPO_URL}/blob/main/DESIGN.md`} className="hover:underline">
              Design system
            </a>
            <a href={`${REPO_URL}/blob/main/LICENSE`} className="hover:underline">
              MIT License
            </a>
          </div>
          <span>Built for the Turing program, UMass Amherst CICS</span>
        </div>
      </footer>
    </div>
  );
}
