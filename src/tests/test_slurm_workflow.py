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
            self.assertIn('NNI_FILE="$HOME/codes/.secrets/nni"', script)
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
            'SOURCE_ROOT="$nni@selena.hpc.edf.fr:/scratch/users/$nni/codes/$PROJECT_NAME"',
            results,
        )
        self.assertIn(
            'SCRATCH_PROJECT_ROOT="/scratch/users/$nni/codes/$PROJECT_NAME"',
            code,
        )
        self.assertIn('"mkdir -p \'$SCRATCH_PROJECT_ROOT/outputs_selena\'', code)
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
        compute_fronts = [front for front in fronts if "_tables" not in front.stem]
        table_fronts = [front for front in fronts if "_tables" in front.stem]
        self.assertFalse(list(ROOT.glob("*.slurm")))
        self.assertEqual(len(dgx_fronts), 18)
        self.assertEqual(len(selena_fronts), 16)
        self.assertEqual(len(list((slurm_root / "dgx/main").glob("*.slurm"))), 2)
        self.assertEqual(len(list((slurm_root / "dgx/ablations").glob("*.slurm"))), 14)
        self.assertEqual(len(list((slurm_root / "dgx/deadline").glob("*.slurm"))), 2)
        self.assertEqual(len(list((slurm_root / "selena/main").glob("*.slurm"))), 2)
        self.assertEqual(
            len(list((slurm_root / "selena/ablations").glob("*.slurm"))), 14
        )
        self.assertEqual(
            {path.stem for path in selena_fronts},
            {
                f"{path.stem}_selena"
                for path in dgx_fronts
                if path.parent.name != "deadline"
            },
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
        for front in compute_fronts:
            text = front.read_text(encoding="utf-8")
            for fragment in required:
                self.assertIn(fragment, text, front.name)
            self.assertNotIn("BASH_SOURCE", text, front.name)

        for front in table_fronts:
            text = front.read_text(encoding="utf-8")
            for fragment in (
                "#!/bin/bash",
                "#SBATCH --gres=gpu:1",
                "#SBATCH --time=01:00:00",
                "#SBATCH --nodes=1",
                "#SBATCH --ntasks=1",
                "#SBATCH --cpus-per-task=4",
                "#SBATCH --mem=16000",
                'PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"',
                "STAGES=tables",
                'cd "$PROJECT_ROOT"',
                'source "$PROJECT_ROOT/src/slurm/run_family.sh"',
            ):
                self.assertIn(fragment, text, front.name)

        for front in dgx_fronts:
            text = front.read_text(encoding="utf-8")
            job_pattern = "%A_%a" if "#SBATCH --array=" in text else "%j"
            self.assertIn(f"#SBATCH --output=logs/%x_{job_pattern}.out", text, front.name)
            self.assertIn(f"#SBATCH --error=logs/%x_{job_pattern}.err", text, front.name)
            self.assertIn("#SBATCH --partition=h100", text, front.name)
            self.assertNotIn("#SBATCH --wckey=", text, front.name)
            if front.parent.name == "deadline":
                continue
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
            job_pattern = "%A_%a" if "#SBATCH --array=" in text else "%j"
            self.assertIn(
                f"#SBATCH --output=/scratch/users/%u/codes/online_adaptation/logs_selena/%x_{job_pattern}.out",
                text,
                front.name,
            )
            self.assertIn(
                f"#SBATCH --error=/scratch/users/%u/codes/online_adaptation/logs_selena/%x_{job_pattern}.err",
                text,
                front.name,
            )
            self.assertIn("#SBATCH --partition=an", text, front.name)
            self.assertIn("#SBATCH --qos=an_preemptable", text, front.name)
            self.assertIn("#SBATCH --exclusive", text, front.name)
            self.assertNotIn("#SBATCH --no-requeue", text, front.name)
            self.assertIn("#SBATCH --wckey=P12CU:DATASCIENCE", text, front.name)
            self.assertIn(
                'source "$PROJECT_ROOT/src/slurm/selena_runtime.sh"', text
            )
            self.assertIn('EXPERIMENT_LAUNCH_ID="selena_${SLURM_', text)

        fixed_deadline = (slurm_root / "dgx/deadline/fixed_ablation_30_50_20.slurm").read_text(encoding="utf-8")
        tsrag_deadline = (slurm_root / "dgx/deadline/tsrag_time_t3.slurm").read_text(encoding="utf-8")
        self.assertNotIn("#SBATCH --array=", fixed_deadline)
        self.assertIn("DATASETS_OVERRIDE='[Electricity,Solar]'", fixed_deadline)
        self.assertIn("RANGES_OVERRIDE='[short,long]'", fixed_deadline)
        self.assertIn("EXPERIMENT_FAMILY=deadline_fixed_protocol", fixed_deadline)
        self.assertIn('QUERY_STRIDE="${QUERY_STRIDE:-127}"', fixed_deadline)
        self.assertIn('STAGES="${STAGES:-extract,adapt,tables}"', fixed_deadline)
        self.assertNotIn("#SBATCH --array=", tsrag_deadline)
        self.assertIn(
            "DATASETS_OVERRIDE='[time/ne_china_wind_h,time/coastal_t_s_h_part11,time/sg_weather_d]'",
            tsrag_deadline,
        )
        self.assertIn("EXPERIMENT_FAMILY=deadline_tsrag_comparison", tsrag_deadline)
        self.assertIn('QUERY_STRIDE="${QUERY_STRIDE:-127}"', tsrag_deadline)
        self.assertIn('STAGES="${STAGES:-extract,adapt,tables}"', tsrag_deadline)

        selena_runtime = (ROOT / "src/slurm/selena_runtime.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('${NNI_FILE:-$HOME/codes/.secrets/nni}', selena_runtime)
        self.assertIn(
            'SELENA_SCRATCH_PROJECT_ROOT="/scratch/users/$selena_nni/codes/$PROJECT_NAME"',
            selena_runtime,
        )
        self.assertIn(
            'OUTPUTS_ROOT="${OUTPUTS_ROOT:-$SELENA_SCRATCH_PROJECT_ROOT/outputs_selena}"',
            selena_runtime,
        )
        self.assertIn(
            'LOGS_ROOT="${LOGS_ROOT:-$SELENA_SCRATCH_PROJECT_ROOT/logs_selena}"',
            selena_runtime,
        )

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
