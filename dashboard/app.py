"""Phase 6 - FedGuard results dashboard (Streamlit).

Run with:  streamlit run dashboard/app.py

Reads results/results.csv (plus results/cloud_results.csv for the Phase 4
cross-cloud run) and renders the three-way comparison the whole project builds
toward: centralized vs. federated vs. federated + differential privacy.
"""

import os
import sys
import time

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV = os.path.join(ROOT, "results", "results.csv")
CLOUD_CSV = os.path.join(ROOT, "results", "cloud_results.csv")

RUN_LABELS = {
    "centralized": "Centralized baseline",
    "federated": "Federated (local sim)",
    "federated_nodp": "Federated, no DP",
    "federated_dp": "Federated + DP",
    "federated_cloud": "Federated, cross-cloud (AWS+GCP)",
    "manual_check": "Phase 5 weight-check run",
}
METRICS = ["accuracy", "precision", "recall", "f1", "auc"]
SOURCE_ORDER = ["centralized", "federated_nodp", "federated_dp",
                "federated", "federated_cloud"]


def load_results() -> pd.DataFrame:
    frames = []
    for path in (RESULTS_CSV, CLOUD_CSV):
        if os.path.exists(path):
            frame = pd.read_csv(path)
            frame["source"] = frame["source"].astype(str)
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["timestamp", "round", "source"] + METRICS + ["epsilon"])
    data = pd.concat(frames, ignore_index=True)
    data = data[data["source"] != "manual_check"]
    for col in METRICS + ["epsilon"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def final_row(data: pd.DataFrame, source: str) -> pd.Series | None:
    rows = data[data["source"] == source]
    if rows.empty:
        return None
    return rows.sort_values("round").iloc[-1]


def main() -> None:
    st.set_page_config(page_title="FedGuard", page_icon="🛡️", layout="wide")
    st.title("FedGuard — Cross-Cloud Federated Learning with Differential Privacy")
    st.caption(f"Live view of `{os.path.relpath(RESULTS_CSV, ROOT)}` — "
               "auto-refreshes every 5 seconds.")

    data = load_results()
    if data.empty:
        st.warning("No results yet — run a phase first "
                   "(`python src/centralized_baseline.py`, then the FL scripts).")
        return

    # ---------------- summary table ----------------
    st.subheader("Summary — final round of each run")
    summary_rows = []
    for source in SOURCE_ORDER + sorted(set(data["source"]) - set(SOURCE_ORDER)):
        row = final_row(data, source)
        if row is None:
            continue
        summary_rows.append({
            "Run": RUN_LABELS.get(source, source),
            "Rounds": int(data[data["source"] == source]["round"].max()),
            "Accuracy": row["accuracy"],
            "Precision": row["precision"],
            "Recall": row["recall"],
            "F1": row["f1"],
            "AUC": row["auc"],
            "ε": row["epsilon"] if pd.notna(row["epsilon"]) else None,
        })
    summary = pd.DataFrame(summary_rows)
    st.dataframe(
        summary.style.format({
            "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
            "F1": "{:.4f}", "AUC": "{:.4f}", "ε": "{:.4f}",
        }, na_rep="—"),
        use_container_width=True, hide_index=True,
    )

    # ---------------- charts ----------------
    left, right = st.columns(2)

    with left:
        st.subheader("Metrics vs. training round")
        metric = st.selectbox("metric", METRICS, index=4, label_visibility="collapsed")
        chart = data.pivot_table(index="round", columns="source", values=metric)
        chart = chart.rename(columns={k: RUN_LABELS.get(k, k) for k in chart.columns})
        if "centralized" in data["source"].unique():
            base = final_row(data, "centralized")
            if base is not None and pd.notna(base[metric]):
                chart["Centralized baseline (reference)"] = base[metric]
        st.line_chart(chart)

    with right:
        st.subheader("Privacy budget (ε) vs. round")
        dp = data[data["source"] == "federated_dp"].sort_values("round")
        if dp.empty:
            st.info("No DP run found in results.csv yet "
                    "(run fl_server/fl_client with `--use-dp`).")
        else:
            st.line_chart(dp.set_index("round")["epsilon"])
            last = dp.iloc[-1]
            st.caption(f"Final ε = **{last['epsilon']:.4f}** at δ = 1e-5 "
                       f"(accuracy {last['accuracy']:.4f}). ε grows with every "
                       "round — that is the budget being spent, not a bug.")

    # ---------------- tradeoff ----------------
    st.subheader("Privacy–utility tradeoff")
    nodp, dp_row = final_row(data, "federated_nodp"), final_row(data, "federated_dp")
    cen = final_row(data, "centralized")
    if nodp is not None and dp_row is not None:
        cols = st.columns(4)
        cols[0].metric("Recall (no DP → DP)",
                       f"{dp_row['recall']:.4f}",
                       f"{dp_row['recall'] - nodp['recall']:+.4f}")
        cols[1].metric("AUC (no DP → DP)",
                       f"{dp_row['auc']:.4f}",
                       f"{dp_row['auc'] - nodp['auc']:+.4f}")
        cols[2].metric("Accuracy (no DP → DP)",
                       f"{dp_row['accuracy']:.4f}",
                       f"{dp_row['accuracy'] - nodp['accuracy']:+.4f}")
        cols[3].metric("ε spent", f"{dp_row['epsilon']:.4f}")
        if cen is not None:
            st.caption(f"Centralized reference: accuracy {cen['accuracy']:.4f}, "
                       f"recall {cen['recall']:.4f}, AUC {cen['auc']:.4f}.")
    else:
        st.info("Need both `federated_nodp` and `federated_dp` runs to show "
                "the tradeoff — re-run Phase 3 with and without `--use-dp`.")

    # ---------------- architecture panel ----------------
    with st.expander("Architecture (what you are looking at)", expanded=False):
        st.markdown(
            """
```
                    ┌──────────────────────────────┐
                    │  Orchestrator (GCP e2-micro) │
                    │  fl_server.py : FedAvg       │
                    │  8.231.92.55:8080            │
                    └───────────▲──────────▲───────┘
              weights only ▲   │          │   ▲ weights only
            (never raw rows)│   │          │   │ (never raw rows)
        ┌───────────────────┴┐ ┌┴──────────┴─┐ ┌──────────────────┐
        │ AWS EC2 t3.micro   │ │ GCP e2-micro│ │ laptop (dev)     │
        │ fl_client node1    │ │ fl_client   │ │ Streamlit        │
        │ 65.1.248.113       │ │ node2       │ │ dashboard        │
        │ own partition CSV  │ │ localhost   │ │ results/*.csv    │
        └────────────────────┘ └─────────────┘ └──────────────────┘
```

- **Two clouds, one model.** The identical `fl_client.py` runs on AWS and GCP;
  only `--partition` and `--server_address` differ. Each VM holds its own
  non-IID slice of the dataset; no row ever leaves its own cloud.
- **What crosses the wire:** ~4,097 float32 weights per round (asserted at
  runtime in `fit()`), never raw transactions.
- **Privacy layer:** Opacus DP-SGD clips per-sample gradients and adds Gaussian
  noise before the update, so even the gradients that leave a node cannot be
  inverted into raw records. ε is reported each round.
- **Phases:** 1 centralized → 2 federated → 3 +DP → 4 cross-cloud →
  5 manual FedAvg check → 6 this dashboard → 7 report.
"""
        )

    time.sleep(5)
    st.rerun()


if __name__ == "__main__":
    sys.exit(main())
