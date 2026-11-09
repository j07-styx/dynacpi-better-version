"""Audit completeness and homology-proxy alignment rules; TRAIN-only labels never scored."""
import json,hashlib,unittest
from pathlib import Path
from dynacpi.balm_public_data_dev import ROOT,get_train_only
from scripts.audit_balm_train_protein_parasail_homology_dev import (
    exact_metrics,Components,CORE_RULES)

class ProteinSWAlignmentAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit=json.loads((ROOT/"research/balm_train_only_smith_waterman_protein_group_audit_20261108.json").read_text(encoding="utf8"))

    def test_local_alignment_matches_and_coverage(self):
        a=exact_metrics("MAVHGGKKL","MAVHGGKKL")
        self.assertAlmostEqual(a["identity"],1.)
        self.assertAlmostEqual(a["shorter_coverage"],1.)
        b=exact_metrics("MAVHGGKKL","MAVHGKKL")
        self.assertGreater(b["identity"],.7)
        self.assertGreater(b["shorter_coverage"],.7)

    def test_train_sequence_coverage_and_pairwise_exhaustiveness(self):
        d=self.audit
        self.assertEqual(d["proteins"],639)
        self.assertEqual(d["pairs_scored"],639*638//2)
        self.assertEqual(d["initial_kmer_groups"],519)
        actual={hashlib.sha256(r["protein"].encode()).hexdigest() for r in get_train_only()}
        for key,mapping in d["sequence_sha256_to_alignment_groups"].items():
            self.assertEqual(set(mapping),actual)
            self.assertEqual(len(set(mapping.values())),d["rules"][key]["connected_components"])

    def test_verified_edges_do_not_cross_new_graph_components(self):
        d=self.audit
        edges=ROOT/"research"/d["edge_file"]
        seen=0
        with edges.open(encoding="utf8") as fh:
            for line in fh:
                e=json.loads(line);seen+=1
                self.assertTrue(e["passes_thresholds"])
                self.assertGreaterEqual(e["identity"],.30)
                self.assertGreaterEqual(e["shorter_coverage"],.80)
                for mode in e["passes_thresholds"]:
                    g=d["sequence_sha256_to_alignment_groups"][mode]
                    self.assertEqual(g[e["protein1_sha256"]],g[e["protein2_sha256"]])
        self.assertEqual(seen,1276)

    def test_old_graph_missed_important_alignments(self):
        d=self.audit
        self.assertGreater(d["rules"]["identity40_cover80"]["alignment_edges_crossing_prior_kmer_groups"],300)
        self.assertEqual(d["rules"]["identity30_cover80"]["connected_components"],271)

    def test_component_union(self):
        c=Components(4)
        c.union(0,1);c.union(2,3);c.union(1,3)
        m=c.groups(["b","a","c","d"])
        self.assertEqual(len(set(m.values())),1)

if __name__=="__main__":unittest.main()
