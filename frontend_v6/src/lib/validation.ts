export type ValidationResult = { ok: true; value: string } | { ok: false; error: string };

const SMILES_CHARS = /^[A-Za-z0-9@+\-[\]()=#$:/\\.%*]+$/;

export function validateSmiles(raw: string): ValidationResult {
  const s = raw.trim();
  if (!s) return { ok: false, error: "Enter a compound SMILES string." };
  if (/\s/.test(s)) return { ok: false, error: "SMILES must not contain whitespace." };
  if (s.length > 500) return { ok: false, error: `SMILES is ${s.length} characters; the limit is 500.` };
  if (!SMILES_CHARS.test(s)) {
    const bad = [...s].find((c) => !SMILES_CHARS.test(c));
    return { ok: false, error: `Unexpected character "${bad}" in SMILES.` };
  }
  let paren = 0;
  let bracket = false;
  for (const c of s) {
    if (c === "[") {
      if (bracket) return { ok: false, error: "Nested '[' in SMILES." };
      bracket = true;
    } else if (c === "]") {
      if (!bracket) return { ok: false, error: "Unmatched ']' in SMILES." };
      bracket = false;
    } else if (!bracket && c === "(") paren++;
    else if (!bracket && c === ")") {
      paren--;
      if (paren < 0) return { ok: false, error: "Unmatched ')' in SMILES." };
    }
  }
  if (bracket) return { ok: false, error: "Unclosed '[' in SMILES." };
  if (paren !== 0) return { ok: false, error: "Unbalanced parentheses in SMILES." };
  // Ring-closure digits outside brackets must pair up.
  const stripped = s.replace(/\[[^\]]*\]/g, "");
  const open = new Set<string>();
  const re = /%(\d{2})|(\d)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(stripped))) {
    const k = m[1] ?? m[2] ?? "";
    if (open.has(k)) open.delete(k);
    else open.add(k);
  }
  if (open.size) return { ok: false, error: `Unclosed ring bond ${[...open].join(", ")}.` };
  return { ok: true, value: s };
}

const AA = /^[ACDEFGHIKLMNPQRSTVWYX]+$/;
export const PROTEIN_MIN = 10;
export const PROTEIN_MAX = 2048;

/** Accepts raw sequence or single-record FASTA; returns the cleaned uppercase sequence. */
export function cleanProtein(raw: string): string {
  return raw
    .split(/\r?\n/)
    .filter((l) => !l.trim().startsWith(">"))
    .join("")
    .replace(/[\s*]/g, "")
    .toUpperCase();
}

export function validateProtein(raw: string): ValidationResult {
  if ((raw.match(/^\s*>/gm) ?? []).length > 1)
    return { ok: false, error: "Paste a single sequence; multiple FASTA records found." };
  const s = cleanProtein(raw);
  if (!s) return { ok: false, error: "Enter a protein amino-acid sequence." };
  if (!AA.test(s)) {
    const bad = [...new Set([...s].filter((c) => !AA.test(c)))].slice(0, 5).join(" ");
    return { ok: false, error: `Non amino-acid characters: ${bad}` };
  }
  if (s.length < PROTEIN_MIN) return { ok: false, error: `Sequence is ${s.length} residues; minimum is ${PROTEIN_MIN}.` };
  if (s.length > PROTEIN_MAX) return { ok: false, error: `Sequence is ${s.length} residues; maximum is ${PROTEIN_MAX}.` };
  return { ok: true, value: s };
}