# DynaCPI v6 React research workspace

This is the existing Lovable Midnight & Violet frontend adapted locally to drive the REAL v6 full-DAVIS neural ensemble. This source copy preserves the existing layout and Three.js molecule viewer.

## Run on the original authorized PC

1. Start the v6 inference service in the parent repository: run serve_full_davis_final_v6.py under the original dynacpi_v4 Python virtual environment (local server port 8783).
2. Keep the previous v5 demo_full_davis_v5.py service running on port 8782 to show its separately held-out evaluation evidence.
3. From the original local frontend directory C:\Users\J0124\DynaCPI-Frontend run npm install, then npm run dev -- --host 127.0.0.1 --port 5173.
4. Open http://127.0.0.1:5173/ and click Load DAVIS example then Predict affinity, or type any valid SMILES and amino-acid sequence.

Vite dev-proxies /api/v2 to localhost:8783 and /api/v1 to localhost:8782. These are LOCAL-ONLY research services and should not be deployed publicly without production hardening.

## Science integrity

- v6 final refit was trained on all 442 original DAVIS protein reference entries (379 distinct sequences) × 68 compounds = 30,056 pairs. Its in-sample macro RMSE is 0.298741 versus 0.737121 chemical-only. This is TRAIN FIT, NOT a held-out accuracy measure.
- v5 separate held-out evaluation used 302 train, 59 validation, 81 held-out protein entries; test macro RMSE 0.585203 versus 0.823596 chemical-only, under approximate protein grouping. These are **different checkpoints** from final v6.
- Inference accepts arbitrary SMILES/sequence, but predictions on unseen chemicals or proteins have no independent accuracy validation. Confidence intervals are not supplied.
- No matched original published PerceiverCPI benchmark.

## Verified checks

TypeScript --noEmit PASS; Vitest 13/13 PASS; Vite production build PASS; frontend HTTP 200; /api/v2 health and training metrics HTTP 200; v6 POST inference through Vite proxy returned a true known original DAVIS pair result and an exploratory novel-ligand result; malformed SMILES rejected HTTP 400. Backend source data, checkpoint binaries and ESM cache are not included in this public source snapshot.
