from typing import Dict, List, Sequence


def plot_metrics_by_seed_png(
    *,
    out_path: str,
    title: str,
    seeds: Sequence[int],
    recall: Sequence[float],
    accuracy: Sequence[float],
    f1: Sequence[float],
) -> None:
    import matplotlib.pyplot as plt

    x = list(range(len(seeds)))
    seed_labels = [str(s) for s in seeds]

    plt.figure(figsize=(12, 6))
    plt.plot(x, recall, marker="o", label="Recall")
    plt.plot(x, accuracy, marker="o", label="Accuracy")
    plt.plot(x, f1, marker="o", label="F1")

    plt.xticks(x, seed_labels, rotation=45, ha="right")
    plt.ylim(0.0, 1.05)
    plt.title(title)
    plt.xlabel("Prompt Seed")
    plt.ylabel("Metric")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_recall_boxplot_png(
    *,
    out_path: str,
    title: str,
    blocks: List[Dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    labels: List[str] = []
    data: List[List[float]] = []

    for b in blocks:
        labels.append(str(b["label"]))
        data.append([float(x) for x in b["recall_list"]])

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylim(0.0, 1.05)
    plt.title(title)
    plt.ylabel("Recall")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

