# Design System — NeuroBlock "Paper Lab"

## Product Context
- **What this is:** Drag-and-drop neural network builder — design, train (live over WebSockets), and export models visually.
- **Who it's for:** High-school students (15–18) in the UMass CICS Turing summer program; taught by university staff. Smart audience, allergic to anything babyish.
- **Space/industry:** Node-canvas tools (ReactFlow) × ML education. Peers: n8n, ComfyUI, Scratch, tldraw.
- **Project type:** Canvas web app (builder is the product; landing page is separate).

## Aesthetic Direction
- **Direction:** Paper Lab — a bright lab notebook. Warm paper, sharp ink, citrus signal.
- **Decoration level:** Intentional — dot-grid paper texture, soft card shadows, nothing else.
- **Mood:** Serious tooling that's fun to touch. Professional but playful; never dev-terminal-dark, never toy-pastel.
- **Mode:** Light only. (This natively solves classroom projection.)
- **Reference research:** Linear (restraint), Scratch (the babyish failure mode to avoid), n8n (canvas-as-hero). Direction chosen against the all-dark category norm.

## Typography
- **Display/Hero:** Bricolage Grotesque 600–700 — technical-playful ink-trap personality; app wordmark, page/panel headings, node titles. Tracking -0.02em.
- **Body/UI:** General Sans 400–600 (Fontshare) — labels, settings, buttons, prose.
- **Data/Code:** IBM Plex Mono 400–500 — ALL live-changing numbers, tensor shapes, param counts, logs, exported code. Always `font-variant-numeric: tabular-nums`.
- **THE RULE:** numbers that change live are always Plex Mono. This single rule makes the app feel like instrumentation.
- **Loading:** Google Fonts (Bricolage Grotesque, IBM Plex Mono) + Fontshare (General Sans) `<link>` tags.
- **Scale:** 11 / 12 / 13 / 15 (base) / 18 / 26 / 34 / 64px.

## Color
- **Approach:** Restrained chrome + expressive validated layer palette.
- **Surfaces:** paper `#F7F4EC` · card/node `#FFFFFF` · raised `#FBFAF6` · canvas `#EEF1EA` (dot grid `#D3D8C8`, 24px pitch)
- **Ink:** text `#20231F` · muted `#697064` · border `#DDE1D4` · border-strong `#B9C1B1`
- **Accent (RESERVED):** citrus `#F26B3A` = alive/learning only — Train button, live metrics, signal pulses, training states. Nothing else gets citrus.
- **Focus:** `#2563EB` blue ring, always visible.
- **Semantic:** success `#2E9D68` · warning ink `#A87F10` · error `#D94C4C` · info `#2563EB`

### Layer palette (canvas nodes) — THE teaching device
**Rule: filled rail = layer learns (trainable params). Hollow rail = transforms only.**
| Layer | Hex | Rail |
|---|---|---|
| Input | `#64707D` | hollow (neutral slate — deliberate low-chroma exception; always text-labeled) |
| Conv2D | `#2D7DD2` | filled |
| MaxPooling | `#0899A8` | hollow |
| Flatten | `#A87F10` | hollow |
| Dense | `#7B5CD6` | filled |
| Dropout | `#C9569C` | hollow |
| Activation | `#7A9E2F` | hollow |
| BatchNorm | `#A855CC` | filled |
| LSTM/GRU | `#0F5E75` | filled |
| Output | `#1F7A4D` | filled — "the finish line" |

Validated (dataviz six-checks, light surface): CVD worst adjacent pair ΔE 29 (target ≥ 12), contrast 10/10 ≥ 3:1. Hues are assigned to layer types permanently — never repainted, never cycled.

### Chart series (training charts)
train-loss `#D85425` · val-loss `#2D7DD2` · accuracy `#2E9D68` (validated: CVD ΔE 70, contrast 3/3). Series always direct-labeled at line ends; identity never color-alone.

## Spacing
- **Base unit:** 8px; **Density:** comfortable.
- **Scale:** 2xs(2) xs(4) sm(8) md(16) lg(24) xl(32) 2xl(48) 3xl(64)

## Layout
- **Approach:** Instrument panel, grid-disciplined. Canvas is the hero.
- **Builder composition:** 48px top bar · 230–260px left layer palette (search + categorized compact rows) · flexible center canvas · 300–340px right config panel · collapsible bottom readout strip (mono ticker: params, compile status, epoch, loss, ws state).
- **Border radius:** sm 5px (chips) · md 8–9px (buttons, inputs) · lg 12px (nodes, cards) · full 999px (tags).
- **Nodes:** ~172–200px wide, white body, 1px border, 5px colored left rail (filled/hollow per rule), name in Bricolage 600, params in Plex Mono muted, tensor-shape chip on top-right edge, labeled ports. Soft shadow `0 1px 2px rgba(32,35,31,.06), 0 4px 14px rgba(32,35,31,.05)` — "stickers on a desk."
- **Wires:** 1.8px, stroked in target-layer hue; every wire labeled with its tensor shape in a mono chip.

## Motion
- **Approach:** Intentional; one signature.
- **Signature — the Ignition Sweep:** on Train, node rails light up in dependency order input→output (60ms stagger), then citrus pulses travel along wires (2.4s loop, `stroke-dasharray` march) for the whole run; each epoch = soft graph shimmer; loss chart draws with a trail.
- **Easing:** enter ease-out · exit ease-in · move ease-in-out.
- **Duration:** micro 50–100ms · short 150–250ms · medium 250–400ms.
- **Budget:** 60fps on school laptops — CSS transforms/opacity only; no filter animation on many nodes at once.

## Error/status language
Errors appear ON the node or wire where they happen (red border + exact message, e.g. `Expected 4D tensor`), never as toast spam. The bottom readout strip is the app's voice.

## Anti-patterns (hard NOs)
Purple gradients · glassmorphism · neon-hacker dark · toy pastels · Bootstrap-blue buttons · centered dashboard composition · decorative blobs · anonymous unlabeled ports · dual-axis charts · citrus used for anything that isn't alive/learning.

## Decisions Log
| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-07-14 | Initial system created | /design-consultation: research (Linear/Scratch/n8n) + Codex & Claude outside voices |
| 2026-07-14 | Light "Paper Lab" over dark "Phosphor Lab" | User call: light + playful, not dev-centric; also projects well in classrooms. Dark preview kept at ~/.gstack/projects/aryamangoenka-DND-Neural-Network/designs/design-system-20260714/ |
| 2026-07-14 | Filled/hollow rail = learns/transforms | Palette itself teaches trainable-parameter concept |
| 2026-07-14 | Palette CVD-validated | dataviz six-checks; LSTM bronze→teal after protan ΔE 3.3 failure |
