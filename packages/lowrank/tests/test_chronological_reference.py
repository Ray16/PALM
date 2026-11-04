"""Synthetic policy tests only; these are not real chronological experiments."""

import pytest
from lowrank.validation import chronological_reference as reference


def test_tied_dates_stay_together_and_counts_are_matched_without_dropping_rows():
    ids = ["a", "b", "c", "d", "e", "f"]
    dates = dict(
        zip(
            ids,
            [
                "2020-01-01",
                "2020-01-01",
                "2020-02-01",
                "2020-03-01",
                "2020-03-01",
                "2020-04-01",
            ],
        )
    )
    assignment, spec = reference(ids, dates, ["2020-01-01", "2020-02-01"])
    assert len(assignment) == 6 and assignment["a"] == assignment["b"] == "train"
    assert assignment["d"] == assignment["e"] == "test"
    assert spec.splits == [2, 1, 3] and spec.epsilon == 0


def test_actual_numeric_test_order_supports_two_fold_reference():
    assignment, spec = reference(
        [1, 2, 3, 4], {1: 10, 2: 20, 3: 20, 4: 30}, [20], names=["train", "test"]
    )
    assert spec.splits == [3, 1] and assignment[2] == assignment[3] == "train"


@pytest.mark.parametrize(
    "values,cutoffs",
    [
        ({1: 1, 2: 2, 3: 3}, [float("nan")]),
        ({1: 1, 2: 2, 3: 3}, [4]),
        ({1: 1, 2: 2}, [1]),
        ({1: "2020-01-01", 2: "2020-02-01T00:00:00Z", 3: "2020-03-01"}, ["2020-01-01"]),
        ({1: 1, 2: "2020-02-01", 3: 3}, [1]),
    ],
)
def test_invalid_or_incomplete_ordering_is_rejected(values, cutoffs):
    with pytest.raises(ValueError):
        reference([1, 2, 3], values, cutoffs, names=["train", "test"])


def test_timezone_aware_dates_are_compared_as_actual_instants():
    assignment, spec = reference(
        [1, 2, 3],
        {
            1: "2020-01-01T00:00:00Z",
            2: "2020-01-02T02:00:00+02:00",
            3: "2020-01-03T00:00:00Z",
        },
        ["2020-01-02T00:00:00Z"],
        names=["train", "test"],
    )
    assert assignment[2] == "train" and spec.splits == [2, 1]
