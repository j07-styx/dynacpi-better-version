"""Report integrity for similarity-memory cold-start experiments and controls."""
import json
import unittest

from dynacpi.balm_public_data_dev import ROOT
from dynacpi.cpi_similarity_memory_dev import MEMORY_SPECS

RULES=("identity30_cover80","identity40_cover80")


class SimilarityMemoryCVReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous={}
        cls.paired={}
        cls.baseline={}
        for rule in RULES:
            cls.previous[rule]=json.loads((ROOT/"research"/
                 f"balm_joint_similarity_memory_sw_{rule}_scaffold_cold_traincv_dev.json").read_text())
            cls.paired[rule]=json.loads((ROOT/"research"/
                 f"balm_joint_similarity_memory_sw_{rule}_scaffold_cold_traincv_dev_paired_controls.json").read_text())
            cls.baseline[rule]=json.loads((ROOT/"research"/
                 f"balm_esm8m_nested_sw_{rule}_murcko_scaffold_cold_traincv_dev.json").read_text())

    def test_no_scaffold_or_alignment_overlap_and_full_coverage(self):
        for rule in RULES:
            report=self.paired[rule]
            self.assertEqual(report["protein_alignment_rule"],rule)
            self.assertEqual(report["train_original_rows"],15538)
            folds=report["folds"]
            self.assertEqual(len(folds),4)
            self.assertEqual(sum(f["heldout_original_rows"] for f in folds),15538)
            self.assertEqual(sum(f["heldout_alignment_components"] for f in folds),
                             report["total_heldout_alignment_components"])
            self.assertTrue(all(f["scaffold_overlap"]==0 for f in folds))

    def test_paired_reproduces_preceding_results(self):
        for rule in RULES:
            prev=self.previous[rule]["out_of_fold"]
            cur=self.paired[rule]["out_of_fold"]
            for m in ("pooled_rmse","component_macro_rmse"):
                self.assertEqual(set(cur[m]),set(prev[m]))
                for name in cur[m]:
                    self.assertAlmostEqual(cur[m][name],prev[m][name],places=8)

    def test_unaugmented_reference_reproduced(self):
        for rule in RULES:
            ref=self.baseline[rule]["out_of_fold"]
            now=self.paired[rule]["out_of_fold"]
            for m in ("pooled_rmse","component_macro_rmse"):
                self.assertAlmostEqual(now[m]["nested_esm8_reference"],
                                       ref[m]["nested_kmer_esm"],places=8)

    def test_new_memory_reduces_pooled_vs_reference_in_both_protocols(self):
        for rule in RULES:
            result=self.paired[rule]["out_of_fold"]
            new="ensemble_plus_memory_joint_esm8_k64"
            self.assertLess(result["pooled_rmse"][new],
                            result["pooled_rmse"]["nested_esm8_reference"])
            boot=result["bootstrap"]
            self.assertEqual(boot["replicates"],5000)
            self.assertLess(
                boot["comparisons"][new+"_minus_nested_esm8_reference"]["pooled_CI95"][1],0)
            control=new+"_minus_ensemble_plus_memory_chemical_only_k64"
            self.assertIn(control,boot["comparisons"])

    def test_memory_specs_predeclared(self):
        for result in self.paired.values():
            self.assertEqual(result["memory_specs"],[vars(spec) for spec in MEMORY_SPECS])
            self.assertTrue(result["paired_chemical_controls"])

if __name__=="__main__":
    unittest.main()
