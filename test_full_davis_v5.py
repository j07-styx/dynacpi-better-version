"""Regression and API smoke tests for the full DAVIS v5 research console.

Local checkpoints required. No training, model selection or new test tuning.
"""
import json
import unittest
from pathlib import Path
import numpy as np
from train_full_davis_v5 import OUT,PREP,SPLIT,RUN,LowRankDynaCPI,metrics,sha
import demo_full_davis_v5 as serving

class TestFullDavisV5(unittest.TestCase):
    def test_frozen_split_and_source(self):
        manifest=json.loads(SPLIT.read_text(encoding="utf-8"))
        with np.load(PREP,allow_pickle=False) as z:
            train=z["train"].astype(int)
            val=z["val"].astype(int);test=z["test"].astype(int)
            groups=z["groups"].astype(int)
            self.assertEqual(z["pfeat"].shape,(442,1280))
            self.assertEqual(z["bits"].shape,(68,2048))
            self.assertEqual(z["y"].shape,(442,68))
        self.assertEqual(len(train),302)
        self.assertEqual(len(val),59)
        self.assertEqual(len(test),81)
        self.assertFalse(set(train)&set(val))
        self.assertFalse(set(train)&set(test))
        self.assertFalse(set(groups[train])&set(groups[val]))
        self.assertFalse(set(groups[train])&set(groups[test]))
        self.assertEqual(sha(PREP),manifest["prepared_features_sha256"])

    def test_frozen_selection_and_test_metrics(self):
        results=json.loads((RUN/"RESULTS.json").read_text())
        choice=json.loads((RUN/"SELECTION_FROZEN.json").read_text())
        verified=json.loads((RUN/"VERIFICATION.json").read_text())
        self.assertEqual(choice["selected_family"],results["selected_variant"])
        self.assertTrue(choice["frozen_on_validation"])
        self.assertAlmostEqual(results["selected_test"]["macro_rmse"],0.5852030569843681,places=8)
        self.assertAlmostEqual(results["baseline_test"]["macro_rmse"],0.8235963300577717,places=8)
        self.assertEqual(verified["n_heldout_targets_better_than_baseline"],68)

    def test_endpoint_loads_actual_checkpoints(self):
        client=serving.app.test_client()
        health=client.get("/api/v1/health")
        self.assertEqual(health.status_code,200)
        self.assertEqual(health.json["checkpoint_count"],2)
        self.assertEqual(health.json["full_pairs"],30056)
        html=client.get("/")
        self.assertEqual(html.status_code,200)
        self.assertIn(b"DynaCPI",html.data)
        samples=client.get("/api/v1/examples").json["examples"]
        self.assertEqual(len(samples),1)
        sample=client.post("/api/v1/predict",json={"demo_id":samples[0]["demo_id"]})
        self.assertEqual(sample.status_code,200)
        self.assertGreaterEqual(sample.json["prediction_pKd"],5.0)
        self.assertEqual(sample.json["demo_id"],samples[0]["demo_id"])
        self.assertEqual(client.post("/api/v1/predict",json={"demo_id":"BAD-ID"}).status_code,400)
        self.assertEqual(client.get("/api/v1/training").status_code,200)
        self.assertEqual(client.get("/api/v1/metrics").status_code,200)

if __name__=="__main__":
    unittest.main(verbosity=2)
