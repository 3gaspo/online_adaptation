"""Static contract checks for every online-adaptation Slurm front."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class SlurmWorkflowTest(unittest.TestCase):
    def test_fronts_and_runner_follow_the_single_task_contract(self) -> None:
        fronts = sorted(ROOT.glob("*.slurm"))
        self.assertEqual(len(fronts), 16)
        required = (
            "#!/bin/bash",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --output=logs/%x_%j.out",
            "#SBATCH --error=logs/%x_%j.err",
            "#SBATCH --time=23:00:00",
            "#SBATCH --partition=h100",
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

        runner = (ROOT / "src/slurm/run_family.sh").read_text(encoding="utf-8")
        self.assertIn('STAGES="${STAGES:-extract,adapt,tables}"', runner)
        self.assertIn('srun --ntasks=1 python -m "$module"', runner)
        self.assertNotIn("RUN_MODE", runner)
        self.assertNotIn("BENCHMARK_PROFILE", runner)
        self.assertNotIn("TEST_MODE", runner)
        self.assertNotIn("sbatch ", runner)
        self.assertNotIn("git ", runner)


if __name__ == "__main__":
    unittest.main()
