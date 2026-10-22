"""Generate numeric inputs for the uniformly measured H100 scaling cases."""
from pathlib import Path
import json
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
D = PAPER / "revision_20261021"


def main():
    gate = json.loads((D / "scaling_validation.json").read_text())
    if gate.get("passed") is not True:
        raise ValueError("H100 scaling validation is incomplete")
    d = pd.read_csv(D / "scaling_summary.csv")
    if len(d) != 10 or set(d.host) != {"lambda13"} or set(d.device) != {"NVIDIA H100 NVL"}:
        raise ValueError("scaling summary mixes coverage or hardware")
    p = d[(d.method == "plain") & (d.n == 2_000_000)].iloc[0]
    c = d[(d.method == "combined") & (d.n == 100_000)].iloc[0]
    values = {"ScalePlainLargeSeconds": f"{p.split_s:.1f}",
        "ScalePlainLargeMemory": f"{p.peak_gpu_gib:.1f}",
        "ScaleCombinedLargeSeconds": f"{c.split_s:.1f}"}
    (PAPER / "scaling_numbers.tex").write_text("\n".join("\\newcommand{\\"+k+"}{"+v+"}" for k,v in values.items())+'\n')
    table = d[["method", "n", "split_s", "peak_gpu_gib", "peak_process_rss_gib"]].copy()
    table.method = table.method.map({"plain": "Plain", "combined": "Combined"})
    table = table.rename(columns={"method": "Setting", "n": "Molecules", "split_s": "Seconds",
        "peak_gpu_gib": "GPU GiB", "peak_process_rss_gib": "Host RSS GiB"})
    body = table.to_latex(escape=True,index=False, float_format="%.2f")
    (PAPER / "scaling_table.tex").write_text("\\begin{table}[htbp]\\centering\\small\n"+body+
        "\\caption{Fresh-process H100 NVL scaling cases at rank 256 and size tolerance 0.01. "
        "Split time excludes feature loading and scoring and follows a 100-entity warm-up. "
        "GPU peak is reset after warm-up; process RSS includes inputs and warm-up. "
        "The combined setting uses exact tiled neighbours. These are single-seed measurements, "
        "not repeated timing distributions.}\\end{table}\n")


if __name__ == "__main__":
    main()
