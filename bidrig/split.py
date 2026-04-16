import random
from typing import Dict, List, Sequence, Tuple, TypeVar

T = TypeVar("T")


def split_6_2_2_stratified(items: Sequence[T], labels: Sequence[int], seed: int) -> Dict[str, List[T]]:
    if len(items) != len(labels):
        raise ValueError("items and labels must have the same length")

    pos = [items[i] for i, y in enumerate(labels) if y == 1]
    neg = [items[i] for i, y in enumerate(labels) if y == 0]

    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)

    pos_train, pos_val, pos_test = _split_counts(len(pos))
    neg_train, neg_val, neg_test = _split_counts(len(neg))

    train = pos[:pos_train] + neg[:neg_train]
    val = pos[pos_train : pos_train + pos_val] + neg[neg_train : neg_train + neg_val]
    test = pos[pos_train + pos_val :] + neg[neg_train + neg_val :]

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return {"train": train, "val": val, "test": test}


def split_fixed_test_balance(
    items: Sequence[T],
    labels: Sequence[int],
    *,
    seed: int,
    test_size: int,
    test_pos: int,
) -> Dict[str, List[T]]:
    if len(items) != len(labels):
        raise ValueError("items and labels must have the same length")
    if test_size <= 0:
        raise ValueError("test_size must be positive")
    if test_pos < 0 or test_pos > test_size:
        raise ValueError("test_pos must be in [0, test_size]")
    if test_size >= len(items):
        raise ValueError("test_size must be smaller than total samples")

    rng = random.Random(seed)

    pos_idx = [i for i, y in enumerate(labels) if y == 1]
    neg_idx = [i for i, y in enumerate(labels) if y == 0]

    if len(pos_idx) < test_pos:
        raise ValueError(f"not enough positives for test_pos={test_pos}, total_pos={len(pos_idx)}")
    if len(neg_idx) < (test_size - test_pos):
        raise ValueError(f"not enough negatives for test_size={test_size}, total_neg={len(neg_idx)}")

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    test_idx = pos_idx[:test_pos] + neg_idx[: (test_size - test_pos)]
    rng.shuffle(test_idx)

    test_set = set(test_idx)
    remaining_idx = [i for i in range(len(items)) if i not in test_set]

    n_total = len(items)
    train_size = round(n_total * 0.6)
    val_size = round(n_total * 0.2)
    if train_size + val_size + test_size != n_total:
        train_size = n_total - val_size - test_size
        if train_size <= 0:
            val_size = n_total - test_size - 1
            train_size = 1

    rem_labels = [labels[i] for i in remaining_idx]
    rem_pos_idx = [remaining_idx[j] for j, y in enumerate(rem_labels) if y == 1]
    rem_neg_idx = [remaining_idx[j] for j, y in enumerate(rem_labels) if y == 0]

    rng.shuffle(rem_pos_idx)
    rng.shuffle(rem_neg_idx)

    rem_total = len(remaining_idx)
    train_pos = round(len(rem_pos_idx) * (train_size / rem_total)) if rem_total else 0
    train_pos = max(0, min(train_pos, len(rem_pos_idx)))
    train_neg = train_size - train_pos
    if train_neg < 0:
        train_neg = 0
        train_pos = train_size
    train_neg = max(0, min(train_neg, len(rem_neg_idx)))

    train_idx = rem_pos_idx[:train_pos] + rem_neg_idx[:train_neg]
    rng.shuffle(train_idx)

    train_set = set(train_idx)
    pool_idx = [i for i in remaining_idx if i not in train_set]
    rng.shuffle(pool_idx)

    if len(train_idx) < train_size:
        need = train_size - len(train_idx)
        train_idx.extend(pool_idx[:need])
        pool_idx = pool_idx[need:]
    elif len(train_idx) > train_size:
        extra = train_idx[train_size:]
        train_idx = train_idx[:train_size]
        pool_idx.extend(extra)
        rng.shuffle(pool_idx)

    val_idx = pool_idx

    rng.shuffle(val_idx)

    train = [items[i] for i in train_idx]
    val = [items[i] for i in val_idx]
    test = [items[i] for i in test_idx]

    if len(test) != test_size:
        raise RuntimeError("test size mismatch")

    return {"train": train, "val": val, "test": test}


def _split_counts(n: int) -> Tuple[int, int, int]:
    train = round(n * 0.6)
    val = round(n * 0.2)
    test = n - train - val
    if test < 0:
        test = 0
        val = n - train
    return train, val, test
