import numpy as np
import pandas as pd

# Load your input file
df = pd.read_csv(
    "/scratch/project/tcr_ml/gnn_release/research_scripts/bulk+sc_scores/t1d_imgt_sample_scores.csv"
)  # <- replace with your file path

# Ensure numeric
df["scores"] = pd.to_numeric(df["scores"], errors="coerce")
df = df.dropna(subset=["scores"])

# Clip to avoid logit overflow
eps = 1e-6
p = np.clip(df["scores"].to_numpy(), eps, 1.0 - eps)

# Logit transform
logits = np.log(p / (1.0 - p))
df["logits"] = logits

# Per source mean of raw scores
mean_scores = df.groupby("source")["scores"].mean().rename("Mean")

# Per source mean of logits then inverse logit
mean_logits = df.groupby("source")["logits"].mean().rename("mean_logit")

inv_logit_mean = 1.0 / (1.0 + np.exp(-mean_logits))
inv_logit_mean = inv_logit_mean.rename("Inv Logit Mean")

# Final dataframe
metric_df = pd.concat([mean_scores, inv_logit_mean], axis=1).reset_index()

# Save
metric_df.to_csv("bulk+sc_scores/metric_score/t1d_imgt_metric_scores.csv", index=False)
