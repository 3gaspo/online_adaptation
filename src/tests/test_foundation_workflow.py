"""Dependency-free guards for the shared foundation-model evaluation surface."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class FoundationWorkflowTest(unittest.TestCase):
    def test_backbone_ablation_excludes_tirex_from_the_default_launch(self):
        profiles = (ROOT / "src/pipeline/profiles.py").read_text(encoding="utf-8")
        block = profiles.split('if family == "backbone_ablation":', 1)[1].split(
            'if family == "sota_backbone_ablation":', 1
        )[0]
        active = "\n".join(
            line for line in block.splitlines() if not line.lstrip().startswith("#")
        )
        for name in ("chronos2", "chronos_bolt", "ts_icl", "tabpfn_ts"):
            self.assertIn(f'"{name}"', active)
        self.assertNotIn('"tirex2"', active)
        self.assertIn('# "tirex2"', block)

    def test_resource_roots_follow_adaptation_order(self):
        runner = (ROOT / "src/slurm/run_family.sh").read_text(encoding="utf-8")
        for candidate in (
            '"$PROJECT_ROOT/$name"',
            '"$PROJECT_ROOT/../$name"',
            '"$PROJECT_ROOT/../../../$name"',
        ):
            self.assertIn(candidate, runner)
        self.assertNotIn('"$PROJECT_ROOT/../../$name"', runner)

    def test_shared_adapters_check_the_flat_cluster_parent(self):
        external = ROOT / "src/external_models"
        for name in ("chronos2.py", "chronos_bolt.py", "ts_icl.py", "tirex2.py", "tabpfn.py"):
            source = (external / name).read_text(encoding="utf-8")
            self.assertIn('project.parent / "weights"', source)

    def test_shared_adapters_and_time_preparation_are_runnable_entry_points(self):
        external = ROOT / "src/external_models"
        for name in ("chronos2.py", "chronos_bolt.py", "ts_icl.py", "tirex2.py", "tabpfn.py"):
            self.assertTrue((external / name).is_file(), name)
        self.assertTrue((ROOT / "src/data/time.py").is_file())
        wrapper = (ROOT / "src/scripts/prepare_time_csv.py").read_text(encoding="utf-8")
        self.assertIn("src.data.time", wrapper)

    def test_weight_mapping_covers_every_shared_adapter(self):
        workflow = (ROOT / "src/pipeline/workflow.py").read_text(encoding="utf-8")
        for path in (
            '"chronos2": "chronos2"',
            '"chronos_bolt": "chronos-bolt-base"',
            '"ts_icl": "tsicl/tsicl-v1.ckpt"',
            '"tirex2": "tirex2"',
            '"tabpfn_ts": "tabpfnts/tabpfn-v2.5-regressor-v2.5_default.ckpt"',
        ):
            self.assertIn(path, workflow)

    def test_foundation_aliases_are_unique_and_context_fallback_is_removed(self):
        factory = (ROOT / "src/model_loading/forecast.py").read_text(encoding="utf-8")
        extraction = (ROOT / "src/proposal/extraction.py").read_text(encoding="utf-8")
        for removed in ('"chronos"', '"chronos-2"', '"chronos-bolt"', '"tabpfn"'):
            self.assertNotIn(removed, factory)
        self.assertNotIn("supports_context", extraction)
        self.assertFalse((ROOT / "src/external_models/chronos.py").exists())

    def test_scientific_and_external_dependencies_flow_inward(self):
        for path in (ROOT / "src/proposal").glob("*.py"):
            source = path.read_text(encoding="utf-8")
            for forbidden in (
                "from src.pipeline",
                "from src.results",
                "from src.visualization",
            ):
                self.assertNotIn(forbidden, source, path.name)
        for path in (ROOT / "src/external_models").rglob("*.py"):
            self.assertNotIn(
                "from src.proposal",
                path.read_text(encoding="utf-8"),
                str(path.relative_to(ROOT)),
            )
        self.assertFalse((ROOT / "src/external_models/tsrag/evaluate.py").exists())
        for name in ("adaptation.py", "extraction.py", "tsrag.py"):
            self.assertTrue((ROOT / "src/pipeline" / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
