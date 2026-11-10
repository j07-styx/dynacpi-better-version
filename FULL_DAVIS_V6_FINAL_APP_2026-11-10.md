# DynaCPI v6 — Full DAVIS final model and frontend integration

**Executed:** 10 November 2026. **Status:** v6 full-data refit trained, independently verified and live through the original Midnight & Violet frontend.

Local links: Frontend http://127.0.0.1:5173/ ; final inference http://127.0.0.1:8783/api/v2/health ; separate old v5 experimental evaluation http://127.0.0.1:8782/api/v1/health .

## Model and training

The model is LowRankDynaCPI, NOT original PerceiverCPI. The final two-seed refit used all canonical original public DeepDTA DAVIS labels: 442 protein reference entries (379 distinct sequences) × 68 compounds = **30,056 pairs**. Frozen ESM-2 t33 650M 1280D mean protein representations and RDKit 2048-bit Morgan fingerprints feed a 32-dimensional interaction bottleneck, a compound-mean prior and a censored-aware training objective (heuristic pKd=5 weak-binder flag).

The two final model seed epochs (29,24) were frozen using v5 validation only, then refitted from scratch on ALL DAVIS labels. The checkpoints are LOCAL ONLY under full_davis_demo_20261109/final_all442_v6/checkpoints/; do not publish raw assay matrices, ESM caches or weights.

## Verified v6 final TRAIN-FIT metrics (not held-out)

The file verify_full_davis_final_v6.py reconstructed full 30,056-pair inference with the original data, hash-verified both seed checkpoints, and saved METRICS_TRAIN_FIT.json locally.

| Metric (lower better) | Chemical-only | Final v6 ensemble |
|---|---:|---:|
| Target-macro RMSE, IN-SAMPLE | 0.7371208455 | **0.2987410402** |
| Pooled RMSE, IN-SAMPLE | 0.7869056373 | **0.3257397043** |
| Centered MSE, IN-SAMPLE | 0.5180478818 | **0.1005214957** |
| Target-bias MSE, IN-SAMPLE | 0.1011726003 | **0.0055848592** |

The final training-fit RMSE reduction relative to the compound-mean baseline is **59.4719%**. Seed training-fit macro RMSEs were 0.308767 and 0.320053; prediction averaging scored 0.298741.

**CRITICAL:** Final v6 saw all original DAVIS labels. Thus there is **NO independently held-out DAVIS test for this exact model**. Its fit RMSE MUST NEVER be advertised as prospective model performance.

## Previous v5 held-out evaluation — DIFFERENT checkpoint set

The previous v5 evaluation trained on 302 protein entries, selected on 59, tested once on 81 approximately protein-grouped entries. All 68 compound identities were warm. Earlier v5 test macro RMSE was **0.58520305698** vs chemical-only **0.82359633006**, a **28.9454%** relative reduction; v5 test pooled RMSE 0.656132 vs 0.871410, centered MSE 0.378817 vs 0.675195. The sequence grouping uses 3-mer similarity, not curated kinase families. **Those v5 scores are NOT results from the all-data v6 checkpoints.** There is no matched original PerceiverCPI baseline.

## Verified live application

- Existing copied Lovable React/TypeScript Midnight & Violet interface and 3D molecule viewer preserved, running at http://127.0.0.1:5173/.
- Inputs now submit actual SMILES and FASTA/sequence values to the FINAL v6 model, not fixed demo IDs, through Vite same-origin proxy /api/v2 to 127.0.0.1:8783.
- Known DAVIS sequence and chemical: real model prediction returned 7.134586 pKd, correctly flagged IN-SAMPLE.
- Previously unseen chemical CCO against DAVIS protein: real model response 5.72646 pKd, clearly marked EXTRAPOLATIVE and unvalidated.
- Previously unseen protein sequence: frozen local ESM-2 650M encoder executed and real model returned a value (not independent evidence of accuracy).
- Invalid molecular SMILES returned HTTP 400.
- GET /api/v2/health, /api/v2/example, /api/v2/metrics and POST /api/v2/predict tested directly and through the frontend proxy.
- Separate old v5 /api/v1 endpoints still function. Frontend displays v6 training-fit numbers separately from v5 held-out evidence.
- TypeScript check PASS, frontend Vitest **13/13 PASS**, production Vite build PASS, live server HTTP 200 and proxy responses HTTP 200.

## Local execution

Final model service: C:\Users\J0124\dynacpi_v4\.venv\Scripts\python.exe C:\Users\J0124\DynaCPI-Backend\serve_full_davis_final_v6.py

Frontend: change directory to C:\Users\J0124\DynaCPI-Frontend, then run npm run dev -- --host 127.0.0.1 --port 5173 .

Open http://127.0.0.1:5173/ . Load DAVIS example for a demonstrated trained pair, click Predict affinity, or paste a different SMILES and protein.

## Limitations

Final v6 fits all labeled pairs, lacks independent validation of unseen protein/compound generalization, uses heuristic censoring and proxy protein groups, and is a different architecture from published PerceiverCPI. The earlier v5 81-protein held-out result is a useful development estimate but cannot be assigned to this final refit. The local API is for research demos only, not clinical decision-making or public deployment.
