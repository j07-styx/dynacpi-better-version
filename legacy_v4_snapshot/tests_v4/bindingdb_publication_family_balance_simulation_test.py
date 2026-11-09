"""Publication-family feasible split simulation invariants and nonleakage tests."""
import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path
import numpy as np

from dynacpi.balm_public_data_dev import ROOT
from scripts.simulate_bindingdb_publication_group_balance_dev import (
    ROLES,FRACTIONS,quotas,simulate_group_task_distributions,
)

OUTPUT=ROOT/"research/bindingdb_publication_family_split_feasibility_simulation_20261108.json"
SOURCE=ROOT/"research/bindingdb_publication_connected_protein_review_preflight_20261108.json"

class BindingDBSourceFamilySimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=json.loads(OUTPUT.read_text(encoding="utf8"))

    def test_quotas_are_integer_cover_all_families(self):
        for n in (25,26,40,142,168):
            a=quotas(n)
            self.assertEqual(set(a),set(ROLES))
            self.assertEqual(sum(a.values()),n)
            self.assertTrue(all(x>=3 for x in a.values()))

    def test_synthetic_data_reproducibility_and_mass_conservation(self):
        rows={f"g{i:02d}":1+i for i in range(25)}
        first=simulate_group_task_distributions(rows,draws=73,seed=999)
        second=simulate_group_task_distributions(dict(reversed(list(rows.items()))),
                                                draws=73,seed=999)
        self.assertEqual(first,second)
        self.assertEqual(first["total_technical_tasks"],sum(rows.values()))
        self.assertTrue(first["no_actual_split_or_task_assignment_written"])
        self.assertEqual(sum(first["family_count_quota_by_role"].values()),25)
        for role in ROLES:
            metrics=first["role_task_count_distributions"][role]
            self.assertLessEqual(metrics["task_count_p05"],metrics["task_count_median"])
            self.assertLessEqual(metrics["task_count_median"],metrics["task_count_p95"])

    def test_publication_groups_are_fewer_and_imbalance_measured(self):
        s=self.report["simulations"]
        protein=s["provisional_protein_MMseqs_groups_only"]
        source=s["publication_PMid_DOI_patent_connected_groups"]
        self.assertEqual(protein["families"],168)
        self.assertEqual(source["families"],142)
        self.assertEqual(source["total_technical_tasks"],1146)
        self.assertEqual(source["largest_family_tasks"],530)
        self.assertAlmostEqual(source["largest_family_task_fraction"],530/1146)
        self.assertEqual(source["monte_carlo_draws"],1000)
        self.assertGreater(source["role_task_count_distributions"]["test"]["task_count_p95"],530)
        self.assertLess(source["role_task_count_distributions"]["test"]["task_count_median"],1146*.1)

    def test_input_integrity_and_no_real_split(self):
        preflight=SOURCE.read_bytes()
        self.assertEqual(
            self.report["input_source_preflight_sha256"],hashlib.sha256(preflight).hexdigest())
        self.assertTrue(self.report["no_actual_split_or_task_assignment_written"])
        self.assertEqual(self.report["scientific_gate_status"],
                         "BLOCKED_UNAPPROVED_NOT_TRAINING_READY")
        self.assertNotIn("task_roles",self.report)
        self.assertNotIn("actual_roles",self.report)
        for model in self.report["simulations"].values():
            self.assertTrue(model["no_actual_split_or_task_assignment_written"])
            self.assertEqual(set(model["role_task_count_distributions"]),set(ROLES))

if __name__=="__main__":unittest.main()
