"""Property-based tests for RandomForestClassifier."""
import numpy as np
from hypothesis import given, strategies as st
from machine_learning_model.supervised.random_forest import RandomForestClassifier


@given(
    X=st.lists(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False, width=16),
            min_size=2,
            max_size=5,
        ),
        min_size=10,
        max_size=30,
    ),
    y=st.lists(st.integers(min_value=0, max_value=3), min_size=10, max_size=30),
)
def test_random_forest_prediction_shape(X, y):
    if len(X) != len(y):
        return
    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=int)
    rf = RandomForestClassifier(n_estimators=5, max_depth=4)
    rf.fit(X_arr, y_arr)
    preds = rf.predict(X_arr)
    assert preds.shape[0] == X_arr.shape[0]
