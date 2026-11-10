// @ts-nocheck -- index-heavy numeric geometry code; inputs are validated molblock data.
// Browser-only: loads RDKit.js (WASM) from a CDN and builds a visual 3D geometry.
// RDKit MinimalLib provides parsing + 2D coordinates; depth is added with a small
// distance-geometry style relaxation. Visual only — never sent to the model.

export interface Atom { el: string; x: number; y: number; z: number }
export interface Bond { a: number; b: number; order: number }
export interface MolGeom { atoms: Atom[]; bonds: Bond[]; source: "rdkit"; formula: string }

const RDKIT_JS = "https://unpkg.com/@rdkit/rdkit@2024.3.5-1.0.0/dist/RDKit_minimal.js";
const RDKIT_WASM = "https://unpkg.com/@rdkit/rdkit@2024.3.5-1.0.0/dist/RDKit_minimal.wasm";

type RDKitModule = {
  get_mol: (s: string) => null | {
    is_valid: () => boolean;
    get_molblock: () => string;
    get_new_coords?: (useCoordGen?: boolean) => string;
    set_new_coords?: (useCoordGen?: boolean) => boolean;
    get_descriptors?: () => string;
    delete: () => void;
  };
};

let rdkitPromise: Promise<RDKitModule> | null = null;

export function loadRDKit(): Promise<RDKitModule> {
  if (rdkitPromise) return rdkitPromise;
  rdkitPromise = new Promise<RDKitModule>((resolve, reject) => {
    const w = window as unknown as { initRDKitModule?: (o: object) => Promise<RDKitModule> };
    const init = () =>
      w.initRDKitModule!({ locateFile: () => RDKIT_WASM }).then(resolve, reject);
    if (w.initRDKitModule) return init();
    const s = document.createElement("script");
    s.src = RDKIT_JS;
    s.async = true;
    s.onload = init;
    s.onerror = () => reject(new Error("RDKit.js failed to load"));
    document.head.appendChild(s);
  }).catch((e) => {
    rdkitPromise = null;
    throw e;
  });
  return rdkitPromise;
}

function parseMolblock(mb: string): { atoms: Atom[]; bonds: Bond[] } {
  const lines = mb.split("\n");
  const counts = lines[3];
  const na = parseInt(counts.slice(0, 3));
  const nb = parseInt(counts.slice(3, 6));
  const atoms: Atom[] = [];
  for (let i = 0; i < na; i++) {
    const l = lines[4 + i];
    atoms.push({ x: parseFloat(l.slice(0, 10)), y: parseFloat(l.slice(10, 20)), z: 0, el: l.slice(31, 34).trim() });
  }
  const bonds: Bond[] = [];
  for (let i = 0; i < nb; i++) {
    const l = lines[4 + na + i];
    bonds.push({ a: parseInt(l.slice(0, 3)) - 1, b: parseInt(l.slice(3, 6)) - 1, order: parseInt(l.slice(6, 9)) });
  }
  return { atoms, bonds };
}

function relax3D(atoms: Atom[], bonds: Bond[]) {
  const n = atoms.length;
  if (n < 2) return;
  // Normalise 2D layout to ~1.5 Å bonds.
  let avg = 0;
  for (const b of bonds) avg += Math.hypot(atoms[b.a].x - atoms[b.b].x, atoms[b.a].y - atoms[b.b].y);
  const scale = bonds.length ? 1.5 / (avg / bonds.length || 1) : 1;
  let seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5);
  for (const a of atoms) { a.x *= scale; a.y *= scale; a.z = rnd() * 0.8; }
  const adj: number[][] = Array.from({ length: n }, () => []);
  for (const b of bonds) { adj[b.a].push(b.b); adj[b.b].push(b.a); }
  const target = new Map<number, number>();
  const key = (i: number, j: number) => (i < j ? i * n + j : j * n + i);
  for (const b of bonds) target.set(key(b.a, b.b), b.order === 2 ? 1.34 : b.order === 3 ? 1.2 : 1.5);
  for (let i = 0; i < n; i++)
    for (const j of adj[i]) for (const k of adj[i]) if (j < k && !target.has(key(j, k))) target.set(key(j, k), 2.5);
  for (let it = 0; it < 250; it++) {
    const step = 0.08;
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++) {
        const A = atoms[i], B = atoms[j];
        const dx = B.x - A.x, dy = B.y - A.y, dz = B.z - A.z;
        const d = Math.hypot(dx, dy, dz) || 1e-3;
        const t = target.get(key(i, j));
        let f = 0;
        if (t !== undefined) f = (d - t) * 0.5;
        else if (d < 3.2) f = (d - 3.2) * 0.15;
        else continue;
        const fx = (dx / d) * f * step, fy = (dy / d) * f * step, fz = (dz / d) * f * step;
        A.x += fx; A.y += fy; A.z += fz; B.x -= fx; B.y -= fy; B.z -= fz;
      }
  }
  let cx = 0, cy = 0, cz = 0;
  for (const a of atoms) { cx += a.x; cy += a.y; cz += a.z; }
  for (const a of atoms) { a.x -= cx / n; a.y -= cy / n; a.z -= cz / n; }
}

export async function buildGeometry(smiles: string): Promise<MolGeom> {
  const rd = await loadRDKit();
  const mol = rd.get_mol(smiles);
  if (!mol || !mol.is_valid()) { mol?.delete(); throw new Error("RDKit could not parse this SMILES."); }
  try {
    mol.set_new_coords?.(true);
    const { atoms, bonds } = parseMolblock(mol.get_molblock());
    if (!atoms.length) throw new Error("No atoms in structure.");
    relax3D(atoms, bonds);
    let formula = "";
    try {
      const d = JSON.parse(mol.get_descriptors?.() ?? "{}");
      if (d.amw) formula = `${Number(d.amw).toFixed(2)} g/mol`;
    } catch { /* ignore */ }
    return { atoms, bonds, source: "rdkit", formula };
  } finally {
    mol.delete();
  }
}

export async function rdkitParses(smiles: string): Promise<boolean | null> {
  try {
    const rd = await loadRDKit();
    const m = rd.get_mol(smiles);
    const ok = !!m && m.is_valid();
    m?.delete();
    return ok;
  } catch {
    return null;
  }
}