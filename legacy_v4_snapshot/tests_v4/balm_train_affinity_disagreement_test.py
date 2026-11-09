"""Tests for TRAIN-only affinity disagreement audit and fold-training filters."""
import json,unittest
from pathlib import Path

from dynacpi.balm_public_data_dev import ROOT
from scripts.ablate_balm_sw_alignment_affinity_disagreement_train_dev import (
    exclude_conflicting_exact_inputs,median_collapse)

class AffinityInputDisagreementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=json.loads(
            (ROOT/"research/balm_train_only_affinity_replicate_disagreement_audit_20261108.json").read_text(encoding="utf8"))

    def test_report_is_train_only_and_spread_reconciles(self):
        r=self.report
        self.assertEqual(r["raw_training_rows"],15538)
        self.assertEqual(r["unique_exact_protein_canonical_isomeric_smiles"],12315)
        self.assertEqual(r["repeated_exact_input_groups"],1305)
        self.assertEqual(r["rows_in_repeated_groups"],4528)
        self.assertEqual(r["conflict_thresholds_log_affinity_units"]["1.0"]["exact_input_groups"],462)
        self.assertEqual(r["conflict_thresholds_log_affinity_units"]["1.0"]["original_rows_in_groups"],2542)

    def test_median_aggregation_and_mean_input_compatibility(self):
        inputs=[
            {"protein":"AAA","smiles":"CCO","label":2.},
            {"protein":"AAA","smiles":"CCO","label":4.},
            {"protein":"AAA","smiles":"CCO","label":100.},
            {"protein":"BBB","smiles":"CCO","label":3.},
        ]
        result=median_collapse(inputs)
        self.assertEqual(len(result),2)
        row=next(r for r in result if r["protein"]=="AAA")
        self.assertEqual(row["label"],4.)

    def test_discordance_removes_all_repeated_inputs_but_not_distinct_protein(self):
        inputs=[
            {"protein":"AAA","smiles":"CCO","label":2.},
            {"protein":"AAA","smiles":"OCC","label":4.},
            {"protein":"AAA","smiles":"CCN","label":1.},
            {"protein":"BBB","smiles":"CCO","label":4.},
        ]
        canon={"CCO":"CCO","OCC":"CCO","CCN":"CCN"}
        result,diag=exclude_conflicting_exact_inputs(inputs,canon,threshold=1.)
        self.assertEqual(diag["discarded_canonical_exact_input_groups"],1)
        self.assertEqual(diag["discarded_original_training_rows"],2)
        self.assertEqual(len(result),2)
        self.assertEqual({r["protein"] for r in result},{"AAA","BBB"})

    def test_no_exclusion_when_repeats_are_consistent(self):
        inputs=[
            {"protein":"AAA","smiles":"CCO","label":4.},
            {"protein":"AAA","smiles":"OCC","label":4.2},
        ]
        result,diag=exclude_conflicting_exact_inputs(
            inputs,{"CCO":"CCO","OCC":"CCO"},threshold=1.)
        self.assertEqual(len(result),2)
        self.assertEqual(diag["discarded_original_training_rows"],0)

if __name__=="__main__":unittest.main()
