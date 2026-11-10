// @ts-nocheck -- index-heavy numeric geometry code; inputs are validated molblock data.
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { Minus, Pause, Play, Plus, RotateCcw } from "lucide-react";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { OrbitMark } from "@/components/OrbitMark";
import { buildGeometry, type MolGeom } from "@/lib/molecule";

// Element palette tuned to the Midnight & Violet system (visual only).
const EL: Record<string, [number, number]> = {
  C: [0xb9b3cc, 0.34], N: [0x8fa8ff, 0.34], O: [0xff8f9e, 0.34], S: [0xf2d27a, 0.4],
  P: [0xffb070, 0.4], F: [0x9ef0c6, 0.3], Cl: [0x7fe0a0, 0.38], Br: [0xd08a6a, 0.42],
  I: [0xc58cff, 0.46], H: [0xf1edfa, 0.22],
};

type Status = "idle" | "loading" | "ready" | "error";

export default function MoleculeViewer({ smiles }: { smiles: string | null }) {
  const host = useRef<HTMLDivElement>(null);
  const api = useRef<{ reset: () => void; zoom: (f: number) => void; toggle: () => void } | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [err, setErr] = useState("");
  const [geom, setGeom] = useState<MolGeom | null>(null);
  const [spinning, setSpinning] = useState(true);

  useEffect(() => {
    if (!smiles) { setGeom(null); setStatus("idle"); return; }
    let live = true;
    setStatus("loading");
    buildGeometry(smiles)
      .then((g) => { if (live) { setGeom(g); setStatus("ready"); } })
      .catch((e: Error) => { if (live) { setGeom(null); setErr(e.message); setStatus("error"); } });
    return () => { live = false; };
  }, [smiles]);

  useEffect(() => {
    const el = host.current;
    if (!el || !geom) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch {
      setErr("WebGL is not available in this browser."); setStatus("error"); return;
    }
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    el.appendChild(renderer.domElement);
    renderer.domElement.setAttribute("aria-hidden", "true");

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 500);
    const group = new THREE.Group();
    scene.add(group);
    scene.add(new THREE.HemisphereLight(0xe8e0ff, 0x1a1626, 1.1));
    const key = new THREE.DirectionalLight(0xffffff, 2.2); key.position.set(5, 8, 10); scene.add(key);
    const rim = new THREE.DirectionalLight(0xa68bfa, 2.4); rim.position.set(-8, -2, -6); scene.add(rim);

    const sphere = new THREE.SphereGeometry(1, 32, 24);
    const cyl = new THREE.CylinderGeometry(0.075, 0.075, 1, 14);
    const mats = new Map<string, THREE.MeshStandardMaterial>();
    const mat = (el: string) => {
      if (!mats.has(el)) mats.set(el, new THREE.MeshStandardMaterial({ color: (EL[el] ?? [0xd8c8ff])[0], roughness: 0.32, metalness: 0.08 }));
      return mats.get(el)!;
    };
    const bondMat = new THREE.MeshStandardMaterial({ color: 0x8a84a0, roughness: 0.45, metalness: 0.1 });
    let radius = 0;
    for (const a of geom.atoms) {
      const m = new THREE.Mesh(sphere, mat(a.el));
      const r = (EL[a.el] ?? [0, 0.38])[1];
      m.scale.setScalar(r); m.position.set(a.x, a.y, a.z); group.add(m);
      radius = Math.max(radius, Math.hypot(a.x, a.y, a.z) + r);
    }
    const up = new THREE.Vector3(0, 1, 0);
    for (const b of geom.bonds) {
      const A = new THREE.Vector3(geom.atoms[b.a].x, geom.atoms[b.a].y, geom.atoms[b.a].z);
      const B = new THREE.Vector3(geom.atoms[b.b].x, geom.atoms[b.b].y, geom.atoms[b.b].z);
      const dir = B.clone().sub(A); const len = dir.length();
      const q = new THREE.Quaternion().setFromUnitVectors(up, dir.clone().normalize());
      const perp = new THREE.Vector3().crossVectors(dir, camera.position.clone().sub(A).add(new THREE.Vector3(0.3, 0.7, 0.1))).normalize().multiplyScalar(0.13);
      const n = b.order === 2 ? 2 : b.order === 3 ? 3 : 1;
      for (let i = 0; i < n; i++) {
        const off = n === 1 ? 0 : (i - (n - 1) / 2) * 2;
        const c = new THREE.Mesh(cyl, bondMat);
        c.scale.set(n > 1 ? 0.7 : 1, len, n > 1 ? 0.7 : 1);
        c.quaternion.copy(q);
        c.position.copy(A).add(B).multiplyScalar(0.5).add(perp.clone().multiplyScalar(off));
        group.add(c);
      }
    }
    // Orbit accent rings
    const ringMat = new THREE.LineBasicMaterial({ color: 0xa68bfa, transparent: true, opacity: 0.22 });
    const rings: THREE.LineLoop[] = [];
    for (let i = 0; i < 2; i++) {
      const pts = Array.from({ length: 160 }, (_, k) => {
        const t = (k / 160) * Math.PI * 2;
        return new THREE.Vector3(Math.cos(t) * (radius + 1.2 + i * 0.6), Math.sin(t) * (radius + 1.2 + i * 0.6), 0);
      });
      const ring = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts), ringMat);
      ring.rotation.set(1.2 + i * 0.5, 0.3 - i * 0.7, 0);
      scene.add(ring); rings.push(ring);
    }

    const dist = Math.max(8, radius * 3.1);
    const home = new THREE.Vector3(0, 0, dist);
    camera.position.copy(home);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; controls.dampingFactor = 0.08; controls.enablePan = false;
    controls.minDistance = radius + 2; controls.maxDistance = dist * 3;
    controls.autoRotate = !reduced && spinning; controls.autoRotateSpeed = 0.9;

    api.current = {
      reset: () => { camera.position.copy(home); controls.target.set(0, 0, 0); controls.update(); },
      zoom: (f) => { camera.position.multiplyScalar(f); controls.update(); },
      toggle: () => { controls.autoRotate = !controls.autoRotate; setSpinning(controls.autoRotate); },
    };
    if (reduced) setSpinning(false);

    const resize = () => {
      const w = el.clientWidth, h = el.clientHeight;
      renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
    };
    resize();
    const ro = new ResizeObserver(resize); ro.observe(el);
    let raf = 0;
    const clock = new THREE.Clock();
    const tick = () => {
      const dt = Math.min(clock.getDelta(), 0.05);
      if (!reduced) rings.forEach((r, i) => (r.rotation.z += dt * (i ? -0.05 : 0.07)));
      controls.update(); renderer.render(scene, camera);
      raf = requestAnimationFrame(tick);
    };
    tick();
    el.dataset.sceneReady = "true";
    return () => {
      cancelAnimationFrame(raf); ro.disconnect(); controls.dispose();
      sphere.dispose(); cyl.dispose(); bondMat.dispose(); ringMat.dispose();
      mats.forEach((m) => m.dispose()); rings.forEach((r) => r.geometry.dispose());
      renderer.dispose(); renderer.domElement.remove(); delete el.dataset.sceneReady; api.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [geom]);

  return (
    <div className="relative h-full min-h-[360px] w-full overflow-hidden rounded-md bg-background grid-bg">
      <div ref={host} className="absolute inset-0 cursor-grab active:cursor-grabbing" />
      {status !== "ready" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 p-8 text-center">
          <OrbitMark active={status === "loading"} />
          <p className="max-w-xs text-sm text-muted-foreground">
            {status === "idle" && "Enter a valid SMILES to render a conformer preview."}
            {status === "loading" && "Parsing structure with RDKit…"}
            {status === "error" && <span className="text-destructive">{err} Prediction is unaffected.</span>}
          </p>
        </div>
      )}
      {status === "ready" && geom && (
        <>
          <div className="pointer-events-none absolute left-3 top-3 flex flex-col gap-1">
            <span className="micro">Conformer preview</span>
            <span className="font-mono text-xs text-foreground/80">
              {geom.atoms.length} heavy atoms · {geom.bonds.length} bonds{geom.formula && ` · ${geom.formula}`}
            </span>
          </div>
          <div className="absolute right-3 top-3 flex gap-1" role="toolbar" aria-label="Viewer controls">
            <IconBtn label="Zoom in" onClick={() => api.current?.zoom(0.85)}><Plus size={14} /></IconBtn>
            <IconBtn label="Zoom out" onClick={() => api.current?.zoom(1.18)}><Minus size={14} /></IconBtn>
            <IconBtn label={spinning ? "Pause rotation" : "Resume rotation"} onClick={() => api.current?.toggle()}>
              {spinning ? <Pause size={13} /> : <Play size={13} />}
            </IconBtn>
            <IconBtn label="Reset view" onClick={() => api.current?.reset()}><RotateCcw size={13} /></IconBtn>
          </div>
        </>
      )}
      <p className="pointer-events-none absolute inset-x-3 bottom-3 font-mono text-[10.5px] leading-relaxed text-muted-foreground">
        Visual only — not a model input. RDKit 2D layout relaxed to approximate depth; not an optimised conformer. Drag to orbit, scroll to zoom.
      </p>
    </div>
  );
}

function IconBtn({ label, onClick, children }: { label: string; onClick: () => void; children: React.ReactNode }) {
  return (
    <button type="button" aria-label={label} title={label} onClick={onClick}
      className="btn btn-ghost h-8 w-8 justify-center bg-card/80 p-0 font-mono text-xs backdrop-blur">
      {children}
    </button>
  );
}