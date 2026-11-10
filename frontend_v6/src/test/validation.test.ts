import { describe, expect, it } from "vitest";
import { validateProtein, validateSmiles } from "@/lib/validation";
import { parsePrediction } from "@/lib/api";

describe("validation", () => {
  it("accepts valid SMILES", () => expect(validateSmiles("CC(=O)NC1=NN=C(S1)S(N)(=O)=O").ok).toBe(true));
  it("rejects unbalanced parens", () => expect(validateSmiles("CC(=O").ok).toBe(false));
  it("rejects unclosed ring", () => expect(validateSmiles("c1ccccc").ok).toBe(false));
  it("accepts FASTA protein", () => expect(validateProtein(">sp|x\nMSHHWGYGKHNGPEHW").ok).toBe(true));
  it("rejects non amino acid chars", () => expect(validateProtein("MSHHWGYGK1234JJ").ok).toBe(false));
  it("rejects short protein", () => expect(validateProtein("MSHHW").ok).toBe(false));
});
describe("parsePrediction", () => {
  it("reads affinity", () => expect(parsePrediction({ affinity: 6.2 }).affinity).toBe(6.2));
  it("throws when no number is returned", () => expect(() => parsePrediction({ ok: true })).toThrow());
});