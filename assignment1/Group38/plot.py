import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("timing.txt", skiprows=1)

P = data[:, 0].astype(int)
M = data[:, 1].astype(int)
times = data[:, 2]

configs = sorted(set(zip(P, M)), key=lambda x: (x[0], x[1]))

grouped_times = []
labels = []

for p, m in configs:
    mask = (P == p) & (M == m)
    grouped_times.append(times[mask])
    labels.append(f"P={p}\nM={m}")

plt.figure(figsize=(12, 6))
plt.boxplot(grouped_times, labels=labels, showmeans=True)

plt.ylabel("Execution Time (seconds)")
plt.title("Execution Time Distribution")
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("boxplot.png", dpi=300)
plt.show()


