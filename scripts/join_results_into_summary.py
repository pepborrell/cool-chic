import pandas as pd

from coolchic.eval.results import parse_hypernet_metrics
from coolchic.utils.paths import RESULTS_DIR, clic20_results

sweep_path = RESULTS_DIR / "exps/copied/delta-hn/from-orange/"

metrics = parse_hypernet_metrics(sweep_path, "clic20-pro-valid", premature=True)
df = pd.DataFrame([s.model_dump() for seq_res in metrics.values() for s in seq_res])
df = df.dropna(axis=1, how="all")  # Drop columns that are all NaN
df = df.sort_values(by=["seq_name", "rate_bpp"])  # Sort so the plots come out right.
df.to_csv(clic20_results / "delta_hypernet.tsv", sep="\t", index=False)
