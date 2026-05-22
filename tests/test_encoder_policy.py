import torch

from src.trader.encoder_policy import (
    EncoderFeatureControllerConfig,
    EncoderFeatureOnlyPolicy,
)


def test_encoder_only_policy_rollout_shapes() -> None:
    config = EncoderFeatureControllerConfig(
        feature_dim=6,
        encoder_dim=4,
        hidden_dim=5,
        max_trade=0.25,
        round_step=0.01,
    )
    features = torch.randn(2, 7, 6)

    rollout = EncoderFeatureOnlyPolicy(config)(features)

    assert rollout.positions.shape == (2, 7)
    assert rollout.deltas.shape == (2, 7)
    assert rollout.encoded.shape == (2, 7, 4)
    assert rollout.final_state.shape == (2, 4)
    assert float(rollout.positions.detach().min()) >= 0.0
    assert float(rollout.positions.detach().max()) <= 1.0
