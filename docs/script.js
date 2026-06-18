/* ============================================================
   Diffusion Consistency RMT — project page interactivity
   ============================================================ */
const $ = (id) => document.getElementById(id);

let MANIFEST = null;
let SPECTRUM = null;

Promise.all([
  fetch("data/manifest.json").then((r) => r.json()),
  fetch("data/spectrum.json").then((r) => r.json()),
]).then(([m, s]) => {
  MANIFEST = m;
  SPECTRUM = s;
  initSeedExplorer();
  initSizeSlider();
  initKappaWidget();
}).catch((e) => console.error("load error", e));

/* ----------------------------------------------------------
   1. Seed explorer
   ---------------------------------------------------------- */
function initSeedExplorer() {
  const se = MANIFEST.seed_explorer;
  const order = MANIFEST.datasets_order;
  const sel = $("ds-select");
  order.forEach((ds) => {
    const o = document.createElement("option");
    o.value = ds;
    o.textContent = se[ds].label;
    sel.appendChild(o);
  });

  const slider = $("seed-slider");
  const label = $("seed-label");
  const img = $("explorer-img");
  const cap = $("explorer-cap");

  function render() {
    const ds = sel.value;
    const info = se[ds];
    const idx = +slider.value;
    const seed = info.seeds[idx];
    img.src = info.path.replace("{seed}", seed);
    label.textContent = seed;
    const arch = info.has_dit ? "UNet-CNN, DiT, and the linear predictor"
                              : "UNet-CNN and the linear predictor";
    cap.textContent =
      `${info.label} · trained on n=${info.n.toLocaleString()} images per split · seed ${seed}. ` +
      `Generated samples (${arch}) from two disjoint splits, with each split's nearest training images.`;
  }

  function setDataset() {
    const info = se[sel.value];
    slider.max = info.seeds.length - 1;
    slider.value = Math.min(+slider.value, slider.max);
    render();
  }

  sel.addEventListener("change", setDataset);
  slider.addEventListener("input", render);
  $("seed-prev").addEventListener("click", () => {
    slider.value = Math.max(0, +slider.value - 1); render();
  });
  $("seed-next").addEventListener("click", () => {
    slider.value = Math.min(+slider.max, +slider.value + 1); render();
  });
  setDataset();
}

/* ----------------------------------------------------------
   2. Size slider (memorization -> renormalization)
   ---------------------------------------------------------- */
function initSizeSlider() {
  const ss = MANIFEST.size_slider;
  const keys = Object.keys(ss);
  const sel = $("size-ds-select");
  keys.forEach((ds) => {
    const o = document.createElement("option");
    o.value = ds;
    o.textContent = ss[ds].label;
    sel.appendChild(o);
  });

  const slider = $("size-slider");
  const label = $("size-label");
  const img = $("size-img");
  const cap = $("size-cap");

  function render() {
    const ds = sel.value;
    const info = ss[ds];
    const n = info.sizes[+slider.value];
    img.src = info.path.replace("{n}", n);
    label.textContent = n.toLocaleString();
    const regime = n <= 1000 ? "memorization regime — samples echo individual training images"
                 : n >= 10000 ? "renormalization regime — generalization, with smoothing toward the mean"
                 : "transition";
    cap.textContent = `${info.label} · ${info.arch} · split 1 · n=${n.toLocaleString()} (${regime}).`;
  }

  function setDataset() {
    const info = ss[sel.value];
    slider.max = info.sizes.length - 1;
    slider.value = Math.min(+slider.value, slider.max);
    render();
  }

  sel.addEventListener("change", setDataset);
  slider.addEventListener("input", render);
  setDataset();
}

/* ----------------------------------------------------------
   3. Kappa widget — solve Silverstein self-consistency in-browser
   kappa - z = gamma * sum_k  kappa * lam_k * w_k / (kappa + lam_k)
   ---------------------------------------------------------- */
function solveKappa(z, lam, w, gamma, guess) {
  let k = guess && guess > 0 ? guess : z;
  for (let it = 0; it < 80; it++) {
    let S = 0, Sp = 0;
    for (let i = 0; i < lam.length; i++) {
      const den = k + lam[i];
      S += (k * lam[i] * w[i]) / den;
      Sp += (lam[i] * lam[i] * w[i]) / (den * den);
    }
    const f = k - z - gamma * S;
    const fp = 1 - gamma * Sp;
    let step = f / fp;
    let kn = k - step;
    if (kn <= 0) kn = k / 2;          // keep positive
    if (Math.abs(kn - k) < 1e-12 * Math.max(1, k)) { k = kn; break; }
    k = kn;
  }
  return k;
}

function kappaCurve(gamma) {
  const lam = SPECTRUM.eigenvalues, w = SPECTRUM.weights;
  // log-spaced sigma^2 grid, solved from large -> small (analytic continuation)
  const xmin = -4, xmax = 2, N = 90;
  const xs = [], ks = [];
  let guess = null;
  for (let j = N; j >= 0; j--) {
    const lx = xmin + (xmax - xmin) * (j / N);
    const z = Math.pow(10, lx);
    const k = solveKappa(z, lam, w, gamma, guess);
    guess = k;
    xs.push(lx); ks.push(k);
  }
  xs.reverse(); ks.reverse();
  return { xs, ks };
}

function initKappaWidget() {
  const svg = $("kappa-svg");
  const slider = $("gamma-slider");
  const gval = $("gamma-val");

  const VBW = 520, VBH = 420;
  const ML = 56, MR = 16, MT = 16, MB = 44;
  const PW = VBW - ML - MR, PH = VBH - MT - MB;
  const XMIN = -4, XMAX = 2, YMIN = -4, YMAX = 2;
  const xpix = (lx) => ML + ((lx - XMIN) / (XMAX - XMIN)) * PW;
  const ypix = (k) => {
    let ly = Math.log10(Math.max(k, 1e-9));
    ly = Math.min(YMAX, Math.max(YMIN, ly));
    return MT + (1 - (ly - YMIN) / (YMAX - YMIN)) * PH;
  };
  const NS = "http://www.w3.org/2000/svg";
  const mk = (tag, attrs) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  };

  const REF = [0.5, 1, 3, 10];                 // faded reference curves
  const COL = { active: "#3b5bdb", ref: "#c2cae6" };

  function pathFor(curve) {
    return curve.xs.map((lx, i) =>
      (i ? "L" : "M") + xpix(lx).toFixed(1) + " " + ypix(curve.ks[i]).toFixed(1)
    ).join(" ");
  }

  function draw() {
    const gamma = Math.pow(10, +slider.value);
    gval.textContent = gamma.toFixed(gamma < 1 ? 2 : 1);
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    // gridlines + ticks
    for (let d = XMIN; d <= XMAX; d++) {
      svg.appendChild(mk("line", { class: "grid", x1: xpix(d), y1: MT, x2: xpix(d), y2: MT + PH }));
      const t = mk("text", { class: "tick-lbl", x: xpix(d), y: MT + PH + 16, "text-anchor": "middle" });
      t.innerHTML = "10<tspan dy='-5' font-size='8'>" + d + "</tspan>";
      svg.appendChild(t);
    }
    for (let d = YMIN; d <= YMAX; d++) {
      svg.appendChild(mk("line", { class: "grid", x1: ML, y1: ypix(Math.pow(10, d)), x2: ML + PW, y2: ypix(Math.pow(10, d)) }));
      const t = mk("text", { class: "tick-lbl", x: ML - 8, y: ypix(Math.pow(10, d)) + 3, "text-anchor": "end" });
      t.innerHTML = "10<tspan dy='-5' font-size='8'>" + d + "</tspan>";
      svg.appendChild(t);
    }
    // axes
    svg.appendChild(mk("line", { class: "axis", x1: ML, y1: MT + PH, x2: ML + PW, y2: MT + PH }));
    svg.appendChild(mk("line", { class: "axis", x1: ML, y1: MT, x2: ML, y2: MT + PH }));
    const xl = mk("text", { class: "axis-lbl", x: ML + PW / 2, y: VBH - 6, "text-anchor": "middle" });
    xl.textContent = "raw noise  σ²";
    svg.appendChild(xl);
    const yl = mk("text", { class: "axis-lbl", x: 14, y: MT + PH / 2, "text-anchor": "middle", transform: `rotate(-90 14 ${MT + PH / 2})` });
    yl.textContent = "renormalized  κ(σ²)";
    svg.appendChild(yl);

    // identity line κ = σ²
    svg.appendChild(mk("path", { class: "idline", d: `M ${xpix(XMIN)} ${ypix(Math.pow(10, XMIN))} L ${xpix(XMAX)} ${ypix(Math.pow(10, XMAX))}` }));
    const idl = mk("text", { class: "klabel", x: xpix(1.1), y: ypix(Math.pow(10, 1.1)) - 6, fill: "#9aa3bd" });
    idl.textContent = "κ = σ²";
    svg.appendChild(idl);

    // reference curves
    REF.forEach((g) => {
      if (Math.abs(Math.log10(g) - +slider.value) < 0.03) return;
      svg.appendChild(mk("path", { class: "kline", style: `stroke:${COL.ref}`, d: pathFor(kappaCurve(g)) }));
    });
    // active curve
    svg.appendChild(mk("path", { class: "kline", style: `stroke:${COL.active}`, d: pathFor(kappaCurve(gamma)) }));
  }

  slider.addEventListener("input", draw);
  draw();
}

/* ----------------------------------------------------------
   Copy BibTeX
   ---------------------------------------------------------- */
document.addEventListener("click", (e) => {
  if (e.target && e.target.id === "copy-bib") {
    navigator.clipboard.writeText($("bibtex-text").textContent).then(() => {
      const b = e.target; const t = b.textContent;
      b.textContent = "Copied ✓";
      setTimeout(() => (b.textContent = t), 1400);
    });
  }
});
