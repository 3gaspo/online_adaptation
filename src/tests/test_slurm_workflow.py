"""Static contract checks for every online-adaptation Slurm front."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class SlurmWorkflowTest(unittest.TestCase):
    def test_cluster_sync_scripts(self):
        code = (ROOT / "sync_code_to_selena.sh").read_text(encoding="utf-8")
        results = (ROOT / "sync_results_to_dgx.sh").read_text(encoding="utf-8")
        for script in (code, results):
            self.assertIn('PROJECT_NAME="$(basename "$PROJECT_ROOT")"', script)
            self.assertIn("sed -n '1p'", script)
        for excluded in (
            ".git/",
            ".venv/",
            ".secrets/",
            "pyproject.toml",
            "uv.lock",
            "datasets/",
            "weights/",
            "outputs/",
            "logs/",
        ):
            self.assertIn(f"--exclude='{excluded}'", code)
        self.assertIn("selena.hpc.edf.fr", code)
        self.assertIn("--delete", code)
        self.assertNotIn("dgx-front.retd.edf.fr", results)
        self.assertIn(
            'SOURCE_ROOT="$nni@selena.hpc.edf.fr:~/codes/$PROJECT_NAME"',
            results,
        )
        self.assertIn('DESTINATION_ROOT="$PROJECT_ROOT"', results)
        self.assertIn('mkdir -p "$DESTINATION_ROOT/outputs_selena"', results)
        self.assertIn("--include='outputs_selena/.gitkeep'", code)
        self.assertIn("--exclude='outputs_selena/***'", code)
        self.assertIn("--include='logs_selena/.gitkeep'", code)
        self.assertIn("--exclude='logs_selena/***'", code)
        self.assertIn('"$SOURCE_ROOT/outputs_selena/"', results)
        self.assertIn('"$SOURCE_ROOT/logs_selena/"', results)
        self.assertIn("pulled from Selena to DGX", results)
        self.assertNotIn("--delete", results)

    def test_fronts_and_runner_follow_the_single_task_contract(self) -> None:
        slurm_root = ROOT / "slurm"
        dgx_fronts = sorted((slurm_root / "dgx").glob("*/*.slurm"))
        selena_fronts = sorted((slurm_root / "selena").glob("*/*.slurm"))
        fronts = dgx_fronts + selena_fronts
        self.assertFalse(list(ROOT.glob("*.slurm")))
        self.assertEqual(len(dgx_fronts), 16)
        self.assertEqual(len(selena_fronts), 16)
        self.assertEqual(len(list((slurm_root / "dgx/main").glob("*.slurm"))), 2)
        self.assertEqual(len(list((slurm_root / "dgx/ablations").glob("*.slurm"))), 14)
        self.assertEqual(len(list((slurm_root / "selena/main").glob("*.slurm"))), 2)
        self.assertEqual(
            len(list((slurm_root / "selena/ablations").glob("*.slurm"))), 14
        )
        self.assertEqual(
            {path.stem for path in selena_fronts},
            {f"{path.stem}_selena" for path in dgx_fronts},
        )
        required = (
            "#!/bin/bash",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --time=23:00:00",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=16",
            "#SBATCH --mem=80000",
            'PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"',
            'cd "$PROJECT_ROOT"',
            'source "$PROJECT_ROOT/src/slurm/run_family.sh"',
        )
        for front in fronts:
            text = front.read_text(encoding="utf-8")
            for fragment in required:
                self.assertIn(fragment, text, front.name)
            self.assertNotIn("BASH_SOURCE", text, front.name)

        for front in dgx_fronts:
            text = front.read_text(encoding="utf-8")
            self.assertIn("#SBATCH --output=logs/%x_%j.out", text, front.name)
            self.assertIn("#SBATCH --error=logs/%x_%j.err", text, front.name)
            self.assertIn("#SBATCH --partition=h100", text, front.name)
            self.assertNotIn("#SBATCH --wckey=", text, front.name)
            pair = (
                slurm_root
                / "selena"
                / front.parent.name
                / f"{front.stem}_selena.slurm"
            )
            pair_text = pair.read_text(encoding="utf-8")
            family = next(
                line for line in text.splitlines() if line.startswith("EXPERIMENT_FAMILY=")
            )
            self.assertIn(family, pair_text, pair.name)
            dgx_job = next(
                line for line in text.splitlines() if line.startswith("#SBATCH --job-name=")
            )
            selena_job = next(
                line
                for line in pair_text.splitlines()
                if line.startswith("#SBATCH --job-name=")
            )
            self.assertNotEqual(dgx_job, selena_job)

        for front in selena_fronts:
            text = front.read_text(encoding="utf-8")
            self.assertIn("#SBATCH --output=logs_selena/%x_%j.out", text, front.name)
            self.assertIn("#SBATCH --error=logs_selena/%x_%j.err", text, front.name)
            self.assertIn("#SBATCH --partition=an", text, front.name)
            self.assertIn("#SBATCH --qos=an_preemptable", text, front.name)
            self.assertIn("#SBATCH --exclusive", text, front.name)
            self.assertNotIn("#SBATCH --no-requeue", text, front.name)
            self.assertIn("#SBATCH --wckey=P12CU:DATASCIENCE", text, front.name)
            self.assertIn('OUTPUTS_ROOT="$PROJECT_ROOT/outputs_selena"', text)
            self.assertIn('LOGS_ROOT="$PROJECT_ROOT/logs_selena"', text)
            self.assertIn('EXPERIMENT_LAUNCH_ID="selena_${SLURM_JOB_ID', text)

        runner = (ROOT / "src/slurm/run_family.sh").read_text(encoding="utf-8")
        self.assertIn('STAGES="${STAGES:-extract,adapt,tables}"', runner)
        self.assertIn('srun --ntasks=1 python -m "$module"', runner)
        self.assertNotIn("RUN_MODE", runner)
        self.assertNotIn("BENCHMARK_PROFILE", runner)
        self.assertNotIn("TEST_MODE", runner)
        self.assertNotIn("sbatch ", runner)
        self.assertNotIn("git ", runner)
        self.assertIn('LOGS_ROOT="${LOGS_ROOT:-$PROJECT_ROOT/logs}"', runner)
        self.assertIn('OUTPUTS_ROOT="${OUTPUTS_ROOT:-$PROJECT_ROOT/outputs}"', runner)
        self.assertIn('"outputs_root=$OUTPUTS_ROOT"', runner)


if __name__ == "__main__":
    unittest.main()
