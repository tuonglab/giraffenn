import matplotlib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from scipy.special import expit

matplotlib.use("Agg")


import matplotlib.pyplot as plt

from graffit.plotting.charts import (
    _as_pdf_path,
    _maybe_hide_xaxis,
    boxplot_individual_sample,
    plot_inv_logit_per_source,
    plot_roc_from_summary,
    scatterplot_individual_sample,
    summarize_and_plot_inv_logit_means,
)


@pytest.fixture(autouse=True)
def _suppress_show(monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)


def test_as_pdf_path_appends_pdf_extension(tmp_path):
    path = tmp_path / "figure.png"
    result = _as_pdf_path(path)
    assert result == path.with_suffix(".pdf")


def test_as_pdf_path_retains_pdf_extension(tmp_path):
    path = tmp_path / "figure.PDF"
    result = _as_pdf_path(path)
    assert result == path.with_suffix(".pdf")


def test_maybe_hide_xaxis_removes_ticks_when_exceeds_max():
    fig, ax = plt.subplots()
    ax.set_xlabel("Label")
    ax.set_xticks([0, 1])
    _maybe_hide_xaxis(num_items=5, max_items=4)
    assert ax.get_xlabel() == ""
    assert list(ax.get_xticks()) == []
    plt.close(fig)


def test_boxplot_individual_sample_requires_out_path_when_saving():
    with pytest.raises(ValueError):
        boxplot_individual_sample([0.1, 0.2], save=True)


def test_scatterplot_individual_sample_saves_pdf(tmp_path):
    out_path = tmp_path / "scatter.png"
    scatterplot_individual_sample([0.1, 0.5, 0.9], save=True, out_path=out_path)
    assert (tmp_path / "scatter.pdf").exists()


def test_plot_inv_logit_per_source_requires_columns():
    df = pd.DataFrame({"sequence": ["s1"], "scores": [0.5]})
    with pytest.raises(ValueError):
        plot_inv_logit_per_source(df)


def test_plot_inv_logit_per_source_returns_expected_summary():
    df = pd.DataFrame(
        {
            "sequence": ["s1", "s2", "s3", "s4"],
            "scores": [0.2, 0.4, 0.7, 0.6],
            "source": ["A", "A", "B", "B"],
        }
    )
    summary = plot_inv_logit_per_source(df)
    eps = 1e-7
    probs = df["scores"].astype(float).clip(eps, 1 - eps)
    logits = np.log(probs / (1 - probs))
    expected = (
        pd.DataFrame({"source": df["source"], "logit": logits})
        .groupby("source", as_index=False)
        .mean()
        .rename(columns={"logit": "mean_logit"})
    )
    expected["inv_logit_mean"] = expit(expected["mean_logit"])
    expected = (
        expected[["source", "inv_logit_mean"]]
        .sort_values("inv_logit_mean")
        .reset_index(drop=True)
    )
    assert_frame_equal(summary, expected, check_exact=False, atol=1e-12, rtol=1e-9)


def test_plot_inv_logit_per_source_requires_out_dir_when_saving():
    df = pd.DataFrame(
        {
            "sequence": ["s1", "s2"],
            "scores": [0.2, 0.4],
            "source": ["A", "A"],
        }
    )
    with pytest.raises(ValueError):
        plot_inv_logit_per_source(df, save=True)


def test_plot_inv_logit_per_source_saves_figures(tmp_path):
    df = pd.DataFrame(
        {
            "sequence": ["s1", "s2", "s3", "s4"],
            "scores": [0.2, 0.4, 0.7, 0.6],
            "source": ["A", "A", "B", "B"],
        }
    )
    out_dir = tmp_path / "plots"
    plot_inv_logit_per_source(df, save=True, out_dir=out_dir)
    assert (out_dir / "inv_logit_boxplot.pdf").exists()
    assert (out_dir / "inv_logit_mean_scatterplot.pdf").exists()


def test_summarize_and_plot_inv_logit_means_requires_columns():
    cancer_df = pd.DataFrame({"sequence": ["c1"], "scores": [0.8], "source": ["S1"]})
    control_df = pd.DataFrame({"sequence": ["n1"], "scores": [0.3]})
    with pytest.raises(ValueError):
        summarize_and_plot_inv_logit_means(cancer_df, control_df)


def test_summarize_and_plot_inv_logit_means_outputs_expected_values():
    cancer_df = pd.DataFrame(
        {
            "sequence": ["c1", "c2", "c3"],
            "scores": [0.9, 0.8, 0.4],
            "source": ["S1", "S1", "S2"],
        }
    )
    control_df = pd.DataFrame(
        {
            "sequence": ["n1", "n2", "n3"],
            "scores": [0.3, 0.2, 0.6],
            "source": ["S1", "S2", "S2"],
        }
    )
    summary_long, summary_wide = summarize_and_plot_inv_logit_means(
        cancer_df, control_df
    )

    def expected_group(df):
        eps = 1e-7
        probs = df["scores"].astype(float).clip(eps, 1 - eps)
        logits = np.log(probs / (1 - probs))
        return (
            pd.DataFrame({"source": df["source"], "logit": logits})
            .groupby("source", as_index=False)
            .mean()
            .assign(inv_logit_mean=lambda t: expit(t["logit"]))[
                ["source", "inv_logit_mean"]
            ]
        )

    expected_cancer = expected_group(cancer_df).set_index("source")["inv_logit_mean"]
    expected_control = expected_group(control_df).set_index("source")["inv_logit_mean"]

    assert set(summary_long["group"]) == {"Cancer", "Control"}
    for source, expected_value in expected_cancer.items():
        actual = summary_wide.loc[summary_wide["source"] == source, "Cancer"].iloc[0]
        assert actual == pytest.approx(expected_value)
    for source, expected_value in expected_control.items():
        actual = summary_wide.loc[summary_wide["source"] == source, "Control"].iloc[0]
        assert actual == pytest.approx(expected_value)


def test_summarize_and_plot_inv_logit_means_requires_out_dir_when_saving(
    cancer_df=None, control_df=None
):
    cancer_df = pd.DataFrame(
        {
            "sequence": ["c1", "c2"],
            "scores": [0.9, 0.8],
            "source": ["S1", "S1"],
        }
    )
    control_df = pd.DataFrame(
        {
            "sequence": ["n1", "n2"],
            "scores": [0.2, 0.3],
            "source": ["S1", "S1"],
        }
    )
    with pytest.raises(ValueError):
        summarize_and_plot_inv_logit_means(cancer_df, control_df, save=True)


def test_summarize_and_plot_inv_logit_means_saves_pdf(tmp_path):
    cancer_df = pd.DataFrame(
        {
            "sequence": ["c1", "c2"],
            "scores": [0.9, 0.8],
            "source": ["S1", "S1"],
        }
    )
    control_df = pd.DataFrame(
        {
            "sequence": ["n1", "n2"],
            "scores": [0.2, 0.3],
            "source": ["S1", "S1"],
        }
    )
    out_dir = tmp_path / "plots"
    summarize_and_plot_inv_logit_means(
        cancer_df, control_df, save=True, out_dir=out_dir
    )
    assert (out_dir / "inv_logit_mean_cancer_vs_control_boxplot.pdf").exists()


def test_plot_roc_from_summary_requires_columns():
    df = pd.DataFrame({"group": ["Cancer"], "other": [0.5]})
    with pytest.raises(ValueError):
        plot_roc_from_summary(df)


def test_plot_roc_from_summary_requires_positive_and_negative():
    df = pd.DataFrame({"group": ["Cancer", "Cancer"], "inv_logit_mean": [0.8, 0.7]})
    with pytest.raises(ValueError):
        plot_roc_from_summary(df)


def test_plot_roc_from_summary_computes_auc_correctly():
    df = pd.DataFrame(
        {
            "group": ["Cancer", "Cancer", "Control", "Control"],
            "inv_logit_mean": [0.9, 0.8, 0.2, 0.1],
        }
    )
    roc_df, auc_value = plot_roc_from_summary(df)
    assert auc_value == pytest.approx(1.0)
    assert roc_df.iloc[0]["fpr"] == 0.0
    assert roc_df.iloc[-1]["tpr"] == 1.0


def test_plot_roc_from_summary_requires_out_path_when_saving():
    df = pd.DataFrame(
        {
            "group": ["Cancer", "Control"],
            "inv_logit_mean": [0.8, 0.2],
        }
    )
    with pytest.raises(ValueError):
        plot_roc_from_summary(df, save=True)


def test_plot_roc_from_summary_saves_pdf(tmp_path):
    df = pd.DataFrame(
        {
            "group": ["Cancer", "Cancer", "Control", "Control"],
            "inv_logit_mean": [0.9, 0.8, 0.2, 0.1],
        }
    )
    out_path = tmp_path / "roc.svg"
    plot_roc_from_summary(df, save=True, out_path=out_path)
    assert (tmp_path / "roc.pdf").exists()
