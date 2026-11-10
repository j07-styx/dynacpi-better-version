import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, fetchV6Health, fetchV6FitMetrics, predictV6 } from "@/lib/api";

afterEach(()=>vi.unstubAllGlobals());

const actual = {
  status:"ok",model:"LowRankDynaCPI final full-DAVIS 2-seed ensemble",
  model_version:"v6-final-all-davis-refit",prediction_pKd:6.41,
  seed_prediction_std:0.08,unit:"pKd",
  protein_in_original_davis:false,compound_in_original_davis:false,
  known_training_pair:false,experimental_extrapolation:true,
  chemical_prior_source:"Morgan Tanimoto kernel estimate",
  protein_embedding_source:"frozen ESM2 650M encoder",
  comparison_scope:"No independent held-out test for this full-data refit",
  notice:"Exploratory extrapolation; no independently validated accuracy.",
  n_training_pairs:30056,
};
function response(body:unknown, status=200) {
  return {ok:status<400,status,statusText:status===200?"OK":"Bad Request",json:async()=>body};
}

describe("v6 real inference API client",()=>{
  it("sends user-provided compound and sequence to the final model, not a fixed demo ID",async()=>{
    const mock=vi.fn().mockResolvedValue(response(actual));
    vi.stubGlobal("fetch",mock);
    const result=await predictV6({smiles:"CCO",sequence:"ACDEFGHIKLMN"});
    expect(result.prediction_pKd).toBe(6.41);
    expect(result.experimental_extrapolation).toBe(true);
    expect(mock).toHaveBeenCalledTimes(1);
    expect(mock.mock.calls[0]?.[0]).toBe("/api/v2/predict");
    const opts=mock.mock.calls[0]?.[1] as RequestInit;
    expect(opts.method).toBe("POST");
    expect(JSON.parse(opts.body as string)).toEqual({smiles:"CCO",sequence:"ACDEFGHIKLMN"});
  });

  it("refuses to display a made-up or malformed prediction",async()=>{
    vi.stubGlobal("fetch",vi.fn().mockResolvedValue(response({status:"ok",prediction_pKd:"6.5"})));
    await expect(predictV6({smiles:"CCO",sequence:"ACDEFGHIKLMN"})).rejects.toBeInstanceOf(ApiError);
  });

  it("shows backend validation failures rather than defaulting to a predicted value",async()=>{
    vi.stubGlobal("fetch",vi.fn().mockResolvedValue(response({error:"Invalid molecular SMILES"},400)));
    await expect(predictV6({smiles:"invalid!",sequence:"ACDEFGHIKLMN"})).rejects.toThrow("Invalid molecular SMILES");
  });

  it("fetches v6 status and VERIFIED IN-SAMPLE metrics from distinct routes",async()=>{
    const f=vi.fn().mockResolvedValue(response({status:"ok",service:"dynacpi-v6-all-davis"}));
    vi.stubGlobal("fetch",f);
    await fetchV6Health();await fetchV6FitMetrics();
    expect(f.mock.calls.map(x=>x[0])).toEqual(["/api/v2/health","/api/v2/metrics"]);
  });
});
