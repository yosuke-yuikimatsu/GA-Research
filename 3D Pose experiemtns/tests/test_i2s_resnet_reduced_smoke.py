import torch

from clifford.algebra.cliffordalgebra import CliffordAlgebra
from src.model import I2S_ResNet


def _make_model(mixing_layer: str) -> I2S_ResNet:
    algebra = CliffordAlgebra((1, 1, 1))
    return I2S_ResNet(
        algebra=algebra,
        hidden_dim=[8],
        pretrained_backbone=False,
        freeze_backbone=True,
        use_positional_encoding=False,
        output_mode="rotation_matrix",
        ga_head_type="reduced",
        ga_head_mixing_layer=mixing_layer,
    )


def test_i2s_resnet_reduced_gp_smoke():
    model = _make_model("gp")
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 3, 3)


def test_i2s_resnet_reduced_mvlinear_smoke():
    model = _make_model("mvlinear")
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 3, 3)


def test_i2s_resnet_reduced_linear_smoke():
    model = _make_model("linear")
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 3, 3)
