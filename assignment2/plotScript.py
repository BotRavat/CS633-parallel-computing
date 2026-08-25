import pandas as pd
import matplotlib.pyplot as plt


file_path = "timing.txt" 
df = pd.read_csv(file_path, sep=r"\s+")


df["Processes"] = df["Processes"].astype(int)
df["DataSize"] = df["DataSize"].astype(int)
df["Time"] = df["Time"].astype(float)


for size in sorted(df["DataSize"].unique()):
    subset = df[df["DataSize"] == size]


    grouped = [group["Time"].values for _, group in subset.groupby("Processes")]
    labels = sorted(subset["Processes"].unique())

    plt.figure()
    plt.boxplot(grouped, tick_labels=labels)
    plt.xlabel("Number of Processes")
    plt.ylabel("Execution Time (s)")
    plt.title(f"Boxplot of Time vs Processes (DataSize = {size})")

    plt.grid()
    plt.show()