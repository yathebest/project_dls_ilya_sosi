"""Generate figures for the presentation from the results_*.json files.

    python run_experiments.py && python run_hybrid.py && python run_shrink.py
    python make_plots.py            -> figures/*.png
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
BLUE, GREEN, RED, GREY = "#2563eb", "#059669", "#dc2626", "#94a3b8"


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fig_shrink():
    rows = load("results_shrink.json")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in rows:
        ax.scatter(r["memory_mb"], r["recall_at_1"], s=70,
                   color=GREEN if r["variant"].startswith("pca") else BLUE)
        ax.annotate(r["variant"], (r["memory_mb"], r["recall_at_1"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("index memory, MB (log)")
    ax.set_ylabel("Recall@1")
    ax.set_title("Iteration 1: shrink Pareto (recall vs memory)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig("figures/shrink_pareto.png", dpi=160)
    plt.close(fig)


def fig_index():
    rows = load("results_experiments.json")
    names = [r["index"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax2 = ax.twinx()
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], [r["recall_at_1"] for r in rows], width=0.4,
           color=BLUE, label="Recall@1")
    ax2.bar([i + 0.2 for i in x], [r["p50_ms"] for r in rows], width=0.4,
            color=RED, label="p50 ms")
    ax.set_xticks(list(x)); ax.set_xticklabels(names)
    ax.set_ylabel("Recall@1", color=BLUE); ax2.set_ylabel("p50 latency, ms", color=RED)
    ax.set_title("Iteration 1: index (Flat vs HNSW vs IVF-PQ)")
    fig.tight_layout(); fig.savefig("figures/index_compare.png", dpi=160)
    plt.close(fig)


def fig_hybrid():
    out = load("results_hybrid.json")
    systems = list(out)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(systems, [out[s]["metrics"]["recall_at_1"] for s in systems],
           color=[GREY, GREY, BLUE, GREEN])
    for i, s in enumerate(systems):
        ax.text(i, out[s]["metrics"]["recall_at_1"] + 0.005,
                f"{out[s]['metrics']['recall_at_1']:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("Recall@1"); ax.set_ylim(0.7, 1.0)
    ax.set_title("Iteration 2: dense -> bm25 -> hybrid -> +rerank")
    fig.tight_layout(); fig.savefig("figures/hybrid_compare.png", dpi=160)
    plt.close(fig)

    # per-category dense vs hybrid+rerank
    cats = sorted(out["dense"]["metrics"]["per_category_recall_at_1"])
    d = [out["dense"]["metrics"]["per_category_recall_at_1"][c] for c in cats]
    h = [out["hybrid+rerank"]["metrics"]["per_category_recall_at_1"][c] for c in cats]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = range(len(cats))
    ax.bar([i - 0.2 for i in x], d, width=0.4, color=GREY, label="dense (Iter 0)")
    ax.bar([i + 0.2 for i in x], h, width=0.4, color=GREEN, label="hybrid+rerank (Iter 2)")
    ax.set_xticks(list(x)); ax.set_xticklabels(cats, rotation=25, ha="right")
    ax.set_ylabel("Recall@1"); ax.legend()
    ax.set_title("Robustness by noise category (AddrLLM taxonomy)")
    fig.tight_layout(); fig.savefig("figures/robustness_by_category.png", dpi=160)
    plt.close(fig)


def fig_neural():
    ch = load("results_char8k.json")["metrics"]["per_category_recall_at_1"]
    st = load("results_st8k.json")["metrics"]["per_category_recall_at_1"]
    ft = load("results_st8k_ft.json")["metrics"]["per_category_recall_at_1"]
    cats = sorted(ch)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = list(range(len(cats)))
    ax.bar([i - 0.27 for i in x], [ch[c] for c in cats], 0.27, color=GREY,
           label="char-n-gram")
    ax.bar(x, [st[c] for c in cats], 0.27, color="#8b9bb4",
           label="e5 off-the-shelf")
    ax.bar([i + 0.27 for i in x], [ft[c] for c in cats], 0.27, color=BLUE,
           label="e5 fine-tuned")
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=25, ha="right")
    ax.set_ylabel("Recall@1"); ax.legend()
    ax.set_title("Fine-tuned encoder wins across categories (real data, unseen addresses)")
    fig.tight_layout(); fig.savefig("figures/neural_vs_char.png", dpi=160)
    plt.close(fig)


def main():
    made = []
    for name, fn in [("shrink", fig_shrink), ("index", fig_index),
                     ("hybrid", fig_hybrid), ("neural", fig_neural)]:
        try:
            fn(); made.append(name)
        except FileNotFoundError as e:
            print(f"skip {name}: {e.filename} missing (run its script first)")
    print("figures ->", ", ".join(os.listdir("figures")))


if __name__ == "__main__":
    main()
