import numpy as np

from src.splits import patient_split, temporal_holdout_split


def test_patient_split_disjoint_and_complete():
    patients = [f"p{i}" for i in range(100)]
    train_set, test_set = patient_split(patients, test_size=0.2, random_state=42)

    assert train_set.isdisjoint(test_set)
    assert len(train_set | test_set) == len(set(patients))
    assert len(test_set) > 0
    assert len(train_set) > 0


def test_temporal_holdout_split_last_years():
    years = np.array([2007, 2007, 2008, 2009, 2010, 2011, 2011])
    train_mask, val_mask = temporal_holdout_split(years, n_val_years=2)

    val_years = set(years[val_mask].tolist())
    assert val_years == {2010, 2011}
    assert np.all(~(train_mask & val_mask))
    assert (train_mask | val_mask).all()