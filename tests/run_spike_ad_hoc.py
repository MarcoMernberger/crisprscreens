from crisprscreens.services.spike_evaluation import (
    evaluate_multiple_mageck_results,
)
import pandas as pd
import tempfile
from pathlib import Path

# Test 1: both present
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "both.tsv"
    df = pd.DataFrame(
        {
            "id": ["SPIKE_POS_1", "SPIKE_NEG_1", "SPIKE_NEUTRAL_1"],
            "pos|fdr": [0.01, 1.0, 1.0],
            "pos|lfc": [2.0, 0.0, 0.0],
            "pos|rank": [1, 3, 2],
            "pos|score": [10, 0, 0],
            "neg|fdr": [1.0, 0.01, 1.0],
            "neg|lfc": [0.0, -2.0, 0.0],
            "neg|rank": [3, 1, 2],
            "neg|score": [0, 10, 0],
        }
    )
    df.to_csv(p, sep="\t", index=False)
    res = evaluate_multiple_mageck_results({"cmp": p}, combine_directions=True)
    both = res[res["direction"] == "both"].iloc[0]
    assert both["n_expected_hits"] == 2
    assert round(float(both["precision"]), 6) == 1.0
    assert round(float(both["recall"]), 6) == 1.0
    assert round(float(both["f1"]), 6) == 1.0
    print("Test both_present: OK")

# Test 2: only pos
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "only_pos.tsv"
    df = pd.DataFrame(
        {
            "id": ["SPIKE_POS_1", "SPIKE_POS_2", "SPIKE_NEUTRAL_1"],
            "pos|fdr": [0.01, 0.02, 1.0],
            "pos|lfc": [2.0, 1.5, 0.0],
            "pos|rank": [1, 2, 3],
            "pos|score": [10, 9, 0],
            "neg|fdr": [1.0, 1.0, 1.0],
            "neg|lfc": [0.0, 0.0, 0.0],
            "neg|rank": [3, 4, 5],
            "neg|score": [0, 0, 0],
        }
    )
    df.to_csv(p, sep="\t", index=False)
    res = evaluate_multiple_mageck_results({"cmp": p}, combine_directions=True)
    pos = res[res["direction"] == "pos"].iloc[0]
    both = res[res["direction"] == "both"].iloc[0]
    assert both["n_expected_hits"] == pos["n_expected_hits"]
    assert round(float(both["precision"]), 6) == round(
        float(pos["precision"]), 6
    )
    assert round(float(both["recall"]), 6) == round(float(pos["recall"]), 6)
    print("Test only_pos: OK")

# Test 3: only neg
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "only_neg.tsv"
    df = pd.DataFrame(
        {
            "id": ["SPIKE_NEG_1", "SPIKE_NEG_2", "SPIKE_NEUTRAL_1"],
            "pos|fdr": [1.0, 1.0, 1.0],
            "pos|lfc": [0.0, 0.0, 0.0],
            "pos|rank": [3, 4, 5],
            "pos|score": [0, 0, 0],
            "neg|fdr": [0.01, 0.02, 1.0],
            "neg|lfc": [-2.0, -1.5, 0.0],
            "neg|rank": [1, 2, 3],
            "neg|score": [10, 9, 0],
        }
    )
    df.to_csv(p, sep="\t", index=False)
    res = evaluate_multiple_mageck_results({"cmp": p}, combine_directions=True)
    neg = res[res["direction"] == "neg"].iloc[0]
    both = res[res["direction"] == "both"].iloc[0]
    assert both["n_expected_hits"] == neg["n_expected_hits"]
    assert round(float(both["precision"]), 6) == round(
        float(neg["precision"]), 6
    )
    assert round(float(both["recall"]), 6) == round(float(neg["recall"]), 6)
    print("Test only_neg: OK")

print("\nAll ad-hoc tests passed")
