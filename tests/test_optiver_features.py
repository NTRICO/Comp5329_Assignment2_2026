import pandas as pd

from src.datasets.optiver_features import (
    build_optiver_stock_features,
    build_optiver_stock_second_features,
)


def test_build_optiver_stock_features_uses_next_bucket_return() -> None:
    rows = []
    for time_id, base_price in [(1, 1.00), (2, 1.01), (3, 1.03)]:
        for second in [0, 1, 2]:
            rows.append(
                {
                    "time_id": time_id,
                    "seconds_in_bucket": second,
                    "bid_price1": base_price,
                    "ask_price1": base_price + 0.001,
                    "bid_price2": base_price - 0.001,
                    "ask_price2": base_price + 0.002,
                    "bid_size1": 100,
                    "ask_size1": 100,
                    "bid_size2": 80,
                    "ask_size2": 120,
                    "stock_id": 0,
                }
            )
    features, realized_returns, time_ids, names = build_optiver_stock_features(pd.DataFrame(rows))

    assert features.shape[0] == 2
    assert realized_returns.shape == (2,)
    assert time_ids.tolist() == [1, 2]
    assert len(names) == features.shape[1]
    assert realized_returns[0] > 0.0
    assert realized_returns[1] > realized_returns[0]


def test_build_optiver_stock_second_features_uses_next_second_within_bucket() -> None:
    rows = []
    for time_id, base_price in [(1, 1.00), (2, 2.00)]:
        for second in [0, 1, 2]:
            price = base_price + 0.01 * second
            rows.append(
                {
                    "time_id": time_id,
                    "seconds_in_bucket": second,
                    "bid_price1": price,
                    "ask_price1": price + 0.001,
                    "bid_price2": price - 0.001,
                    "ask_price2": price + 0.002,
                    "bid_size1": 100,
                    "ask_size1": 100,
                    "bid_size2": 80,
                    "ask_size2": 120,
                    "stock_id": 0,
                }
            )

    features, realized_returns, time_ids, seconds, names = build_optiver_stock_second_features(
        pd.DataFrame(rows),
        max_time_ids=None,
        seconds_per_bucket=3,
    )

    assert features.shape[0] == 4
    assert realized_returns.shape == (4,)
    assert time_ids.tolist() == [1, 1, 2, 2]
    assert seconds.tolist() == [0, 1, 0, 1]
    assert len(names) == features.shape[1]
    assert (realized_returns > 0.0).all()
    assert (realized_returns < 0.02).all()
