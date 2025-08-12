"""Property-based tests for DecisionTreeClassifier using Hypothesis."""
import numpy as np
from hypothesis import given, strategies as st
from machine_learning_model.supervised.decision_tree import DecisionTreeClassifier


@given(
    X=st.lists(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False, width=16),
            min_size=2,
            max_size=6,
        ),
        min_size=5,
        max_size=20,
    ),
    y=st.lists(st.integers(min_value=0, max_value=2), min_size=5, max_size=20),
)
def test_tree_pure_node_prediction_length(X, y):
    if len(X) != len(y):
        return
    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=int)
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X_arr, y_arr)
    preds = clf.predict(X_arr)
    assert preds.shape[0] == X_arr.shape[0]
