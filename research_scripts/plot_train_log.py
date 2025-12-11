import re

import matplotlib.pyplot as plt

# -------------------------------------------------------
# User settings
file1 = "/scratch/project/tcr_ml/gnn_release/research_scripts/train2.out"
file2 = "/scratch/project/tcr_ml/gnn_release/research_scripts/boltz_train.out"

label1 = "Alphafold"
label2 = "Boltz"

output_pdf = "training_comparison_single_plot.pdf"
# -------------------------------------------------------


def parse_train_file(path):
    epochs, losses, accs = [], [], []
    pattern = re.compile(r"epoch=(\d+)\s+loss=([0-9.]+)\s+acc=([0-9.]+)")
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                # print("Matched line:", line.strip(), "->", m.groups())
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                accs.append(float(m.group(3)))
    print(path, "parsed", len(epochs), "epochs")
    return epochs, losses, accs


# Load data
e1, l1, a1 = parse_train_file(file1)
e2, l2, a2 = parse_train_file(file2)

# Create a single plot
fig, ax1 = plt.subplots(figsize=(10, 5))

# Left axis for loss
ax1.plot(e1, l1, label=f"{label1} Loss", linestyle="-")
ax1.plot(e2, l2, label=f"{label2} Loss", linestyle="--")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True)

# Right axis for accuracy
ax2 = ax1.twinx()
ax2.plot(e1, a1, label=f"{label1} Acc", color="tab:green")
ax2.plot(e2, a2, label=f"{label2} Acc", color="tab:red")
ax2.set_ylabel("Accuracy")

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

plt.title("Training Loss and Accuracy (Single Plot)")
plt.tight_layout()

# Save PDF
plt.savefig(output_pdf)
plt.show()

print(f"Saved figure to {output_pdf}")
