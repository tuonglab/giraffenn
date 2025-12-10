import pandas as pd

# Load the old first file
df1 = pd.read_csv(
    "/scratch/project/tcr_ml/gnn_release/research_scripts/bulk+sc_scores/sle_imgt_sample_scores.csv"
)

# Load the second file
df2 = pd.read_csv(
    "/scratch/project/tcr_ml/gnn_release/research_scripts/bulk+sc_scores/temp_sle_sample_scores.csv"
)

# Concatenate second into first
df1 = pd.concat([df1, df2], ignore_index=True)

# Save back to the old first file
df1.to_csv(
    "/scratch/project/tcr_ml/gnn_release/research_scripts/bulk+sc_scores/sle_imgt_sample_scores.csv",
    index=False,
)
