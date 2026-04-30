# Original code : https://pin.it/5fZhohFry

import torch
import torch.nn as nn
import torchvision
from clifford.models.modules.gp import SteerableGeometricProductLayer
from clifford.models.modules.mvsilu import MVSiLU
from clifford.models.modules.fcgp import FullyConnectedSteerableGeometricProductLayer
from clifford.models.modules.linear import MVLinear
from image_encoders import build_encoder
from image2sphere.so3_utils import so3_healpix_grid, flat_wigner, nearest_rotmat
from e3nn import o3
from typing import List, Union


def _unit_quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    if q.ndim != 2 or q.shape[-1] != 4:
        raise ValueError(f"Expected quaternion shape [B, 4], got {tuple(q.shape)}")
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(dim=-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    return torch.stack(
        [
            torch.stack((ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)), dim=-1),
            torch.stack((2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)), dim=-1),
            torch.stack((2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz), dim=-1),
        ],
        dim=-2,
    )


def _project_multivector_to_rotor(mv: torch.Tensor) -> torch.Tensor:
    if mv.ndim != 2 or mv.shape[-1] != 8:
        raise ValueError(f"Expected multivector shape [B, 8], got {tuple(mv.shape)}")
    rotor = mv[:, [0, 4, 5, 6]]
    rotor = rotor / rotor.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return rotor


def _mv_to_rotation_matrix(mv: torch.Tensor) -> torch.Tensor:
    if mv.ndim != 2 or mv.shape[-1] != 8:
        raise ValueError(f"Expected multivector shape [B, 8], got {tuple(mv.shape)}")

    v0 = mv[:, 0]
    v4 = mv[:, 4]
    v5 = mv[:, 5]
    v6 = mv[:, 6]

    norm = torch.sqrt((v0 * v0 + v4 * v4 + v5 * v5 + v6 * v6).clamp_min(1e-8))
    r0 = v0 / norm
    r4 = v4 / norm
    r5 = v5 / norm
    r6 = v6 / norm

    m = torch.zeros((mv.shape[0], 3, 3), dtype=mv.dtype, device=mv.device)
    m[:, 0, 0] = 1.0 - 2.0 * (r4 * r4 + r5 * r5)
    m[:, 0, 1] = 2.0 * (r0 * r4 - r5 * r6)
    m[:, 0, 2] = 2.0 * (r0 * r5 + r4 * r6)
    m[:, 1, 0] = -2.0 * (r0 * r4 + r5 * r6)
    m[:, 1, 1] = 1.0 - 2.0 * (r4 * r4 + r6 * r6)
    m[:, 1, 2] = 2.0 * (r0 * r6 - r4 * r5)
    m[:, 2, 0] = 2.0 * (r4 * r6 - r0 * r5)
    m[:, 2, 1] = 2.0 * (r0 * r6 + r4 * r5)
    m[:, 2, 2] = 1.0 - 2.0 * (r6 * r6 + r5 * r5)
    return m


def _so3_num_fourier_coeffs(lmax: int) -> int:
    return sum([(2 * l + 1) ** 2 for l in range(lmax + 1)])


def _move_tensors_in_object(obj, device: torch.device, visited: set):
    obj_id = id(obj)
    if obj_id in visited:
        return obj
    visited.add(obj_id)

    if isinstance(obj, torch.Tensor):
        return obj if obj.device == device else obj.to(device)

    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            obj[key] = _move_tensors_in_object(value, device, visited)
        return obj

    if isinstance(obj, list):
        for idx, value in enumerate(obj):
            obj[idx] = _move_tensors_in_object(value, device, visited)
        return obj

    if isinstance(obj, tuple):
        return tuple(_move_tensors_in_object(value, device, visited) for value in obj)

    if hasattr(obj, "__dict__"):
        for attr_name, attr_value in list(vars(obj).items()):
            moved_value = _move_tensors_in_object(attr_value, device, visited)
            if moved_value is not attr_value:
                setattr(obj, attr_name, moved_value)
        return obj

    return obj


def _move_unregistered_tensors_to_device(module: nn.Module, device: torch.device):
    visited = set()
    for submodule in module.modules():
        registered_params = set(dict(submodule.named_parameters(recurse=False)).keys())
        registered_buffers = set(dict(submodule.named_buffers(recurse=False)).keys())

        for attr_name, attr_value in list(submodule.__dict__.items()):
            if attr_name in registered_params or attr_name in registered_buffers:
                continue
            moved_value = _move_tensors_in_object(attr_value, device, visited)
            if moved_value is not attr_value:
                setattr(submodule, attr_name, moved_value)



class I2S(nn.Module):
    def __init__(
        self,
        algebra,
        lmax: int = 6,
        rec_level: int = 3,
        n_mv: int = 8,
        hidden_dim: List = [32],
        temperature: float = 1.0,
        encoder_type: str = "resnet",
    ):
        super().__init__()
        self.algebra = algebra
        self.lmax = int(lmax)
        self.rec_level = int(rec_level)
        self.temperature = float(temperature)

        self.encoder = build_encoder(encoder_type)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        enc_channels = getattr(self.encoder, "output_shape", None)[0]

        self._mv_dim = int(2**algebra.dim)
        self._n_mv = int(n_mv)

        self.project = nn.Linear(enc_channels, self._n_mv * self._mv_dim)

        self.num_coeffs = _so3_num_fourier_coeffs(self.lmax)
        self.ga_head = TralaleroTralala(
            algebra=algebra,
            in_features=self._n_mv,
            hidden_dim=hidden_dim,
            out_features=self.num_coeffs,
        )

        xyx = so3_healpix_grid(rec_level=self.rec_level)
        wign = flat_wigner(self.lmax, *xyx)
        self.register_buffer("so3_xyx", xyx, persistent=False)
        self.register_buffer("so3_wigner_T", wign.transpose(0, 1).contiguous(), persistent=False)
        self.register_buffer("so3_rotmats_cache",o3.angles_to_matrix(*self.so3_xyx),persistent=False)
        

    def forward(self, x: torch.tensor) -> torch.Tensor:
        fmap = self.encoder(x)
        fmap = self.avgpool(fmap).flatten(1)
        mv = self.project(fmap).view(fmap.shape[0], self._n_mv, self._mv_dim)
        coeffs_mv = self.ga_head(mv)
        coeffs = coeffs_mv[..., 0]
        logits = self.logits_on_grid(coeffs)
        logits = logits / max(self.temperature, 1e-8)
        return logits

    def logits_on_grid(self, coeffs: torch.Tensor) -> torch.Tensor:
        if coeffs.dim() == 3:
            coeffs = coeffs.squeeze(1)
        return torch.matmul(coeffs, self.so3_wigner_T)  

    @torch.no_grad()
    def probs_on_grid(self, logits : torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict_rotmat(self, coeffs: torch.Tensor) -> torch.Tensor:
        probs = self.probs_on_grid(coeffs)
        idx = torch.argmax(probs, dim=-1)
        return idx
    
    @torch.no_grad()
    def predict(self, x) : 
        idx = self.predict_rotmat(self.forward(x))
        return self.so3_rotmats_cache[idx]
    
    @torch.no_grad()
    def get_nearest_idx(self, rot_gt : torch.Tensor) :
        return nearest_rotmat(rot_gt,self.so3_rotmats_cache)

    def compute_loss(self, img: torch.Tensor, rot_gt: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        logits = self.forward(img)
        idx = self.get_nearest_idx(rot_gt).long().view(-1)
        return criterion(logits, idx)


class GA_I2S(nn.Module):
    def __init__(
        self,
        algebra,
        lmax: int = 6,
        rec_level: int = 3,
        n_mv: int = 8,
        hidden_dim: List = [32],
        temperature: float = 1.0,
        encoder_type: str = "resnet",
        ga_pool_hw: tuple = (28, 28),
    ):
        super().__init__()
        self.algebra = algebra
        self.lmax = int(lmax)
        self.rec_level = int(rec_level)
        self.temperature = float(temperature)

        if len(ga_pool_hw) != 2:
            raise ValueError("ga_pool_hw must have exactly two elements: (height, width)")
        self.ga_pool_hw = (int(ga_pool_hw[0]), int(ga_pool_hw[1]))
        if self.ga_pool_hw[0] <= 0 or self.ga_pool_hw[1] <= 0:
            raise ValueError("ga_pool_hw values must be positive")

        self.pre_encode_pool = nn.AdaptiveAvgPool2d(self.ga_pool_hw)

        # Keep API compatibility with I2S but force canonical GA encoder.
        _ = encoder_type
        self.encoder = build_encoder("ga_canonical")

        enc_shape = getattr(self.encoder, "output_shape", None)
        if enc_shape is None or len(enc_shape) != 3:
            raise ValueError("GA encoder must expose output_shape = (mv_dim, h, w)")

        self._mv_dim = int(enc_shape[0])
        _ = n_mv
        self._n_mv = int(self.ga_pool_hw[0] * self.ga_pool_hw[1])

        if self._mv_dim != int(2**algebra.dim):
            raise ValueError(
                f"Encoder multivector dim ({self._mv_dim}) must match algebra dim ({2**algebra.dim})"
            )
        self.num_coeffs = _so3_num_fourier_coeffs(self.lmax)
        self.ga_head = TralaleroTralala(
            algebra=algebra,
            in_features=self._n_mv,
            hidden_dim=hidden_dim,
            out_features=self.num_coeffs,
        )

        xyx = so3_healpix_grid(rec_level=self.rec_level)
        wign = flat_wigner(self.lmax, *xyx)
        self.register_buffer("so3_xyx", xyx, persistent=False)
        self.register_buffer("so3_wigner_T", wign.transpose(0, 1).contiguous(), persistent=False)
        self.register_buffer("so3_rotmats_cache", o3.angles_to_matrix(*self.so3_xyx), persistent=False)

    def forward(self, x: torch.tensor) -> torch.Tensor:
        pooled_x = self.pre_encode_pool(x)
        mv_grid = self.encoder(pooled_x)
        b, mv_dim, h, w = mv_grid.shape

        mv = mv_grid.permute(0, 2, 3, 1).reshape(b, h * w, mv_dim)

        coeffs_mv = self.ga_head(mv)
        coeffs = coeffs_mv[..., 0]
        logits = self.logits_on_grid(coeffs)
        logits = logits / max(self.temperature, 1e-8)
        return logits

    def logits_on_grid(self, coeffs: torch.Tensor) -> torch.Tensor:
        if coeffs.dim() == 3:
            coeffs = coeffs.squeeze(1)
        return torch.matmul(coeffs, self.so3_wigner_T)

    @torch.no_grad()
    def probs_on_grid(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict_rotmat(self, coeffs: torch.Tensor) -> torch.Tensor:
        probs = self.probs_on_grid(coeffs)
        idx = torch.argmax(probs, dim=-1)
        return idx

    @torch.no_grad()
    def predict(self, x):
        idx = self.predict_rotmat(self.forward(x))
        return self.so3_rotmats_cache[idx]

    @torch.no_grad()
    def get_nearest_idx(self, rot_gt: torch.Tensor):
        return nearest_rotmat(rot_gt, self.so3_rotmats_cache)

    def compute_loss(self, img: torch.Tensor, rot_gt: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        logits = self.forward(img)
        idx = self.get_nearest_idx(rot_gt).long().view(-1)
        return criterion(logits, idx)

class TralaleroTralala(nn.Module):
    def __init__(
        self,
        algebra,
        in_features: int = 512,
        hidden_dim: Union[int, List[int]] = 32,
        out_features: int = 9,
    ):
        super().__init__()

        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim]
        else:
            hidden_dims = list(hidden_dim)

        if len(hidden_dims) == 0:
            raise ValueError("hidden_dim must be a non-empty int or List[int]")

        self.blocks = nn.ModuleList()
        prev = in_features

        for hd in hidden_dims:
            self.blocks.append(
                nn.ModuleDict({
                    "fc": FullyConnectedSteerableGeometricProductLayer(
                        algebra, in_features=prev, out_features=hd
                    ),
                    "act1": MVSiLU(algebra, hd),
                    "gp": SteerableGeometricProductLayer(algebra, hd),
                    "act2": MVSiLU(algebra, hd),
                })
            )
            prev = hd

        self.out = FullyConnectedSteerableGeometricProductLayer(
            algebra, in_features=prev, out_features=out_features
        )

    def forward(self, x):
        if getattr(self, "_extra_tensors_device", None) != x.device:
            _move_unregistered_tensors_to_device(self, x.device)
            self._extra_tensors_device = x.device

        for b in self.blocks:
            x = b["fc"](x)
            x = b["act1"](x)
            x = b["gp"](x)
            x = b["act2"](x)
        x = self.out(x)
        return x



class TransformerLikeMVBlock(nn.Module):
    def __init__(self, algebra, n_multivectors: int):
        super().__init__()
        if int(n_multivectors) <= 0:
            raise ValueError("n_multivectors must be positive")
        n_multivectors = int(n_multivectors)

        self.attn_like = FullyConnectedSteerableGeometricProductLayer(
            algebra,
            in_features=n_multivectors,
            out_features=n_multivectors,
        )
        self.up = MVLinear(algebra, n_multivectors, 4 * n_multivectors)
        self.act = MVSiLU(algebra, 4 * n_multivectors)
        self.down = MVLinear(algebra, 4 * n_multivectors, n_multivectors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended = self.attn_like(x)
        x_ffn = self.up(attended)
        x_ffn = self.act(x_ffn)
        x_ffn = self.down(x_ffn)
        return attended + x_ffn


class TransformerLikeMVHead(nn.Module):
    def __init__(
        self,
        algebra,
        in_features: int = 512,
        hidden_dim: Union[int, List[int]] = 32,
        out_features: int = 9,
    ):
        super().__init__()

        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim]
        else:
            hidden_dims = list(hidden_dim)

        if len(hidden_dims) == 0:
            raise ValueError("hidden_dim must be a non-empty int or List[int]")

        self.blocks = nn.ModuleList(
            [TransformerLikeMVBlock(algebra, n_multivectors=in_features) for _ in hidden_dims]
        )
        self.out = MVLinear(algebra, in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_extra_tensors_device", None) != x.device:
            _move_unregistered_tensors_to_device(self, x.device)
            self._extra_tensors_device = x.device

        for block in self.blocks:
            x = block(x)
        return self.out(x)


class ReducedGeometricProductHead(nn.Module):
    def __init__(
        self,
        algebra,
        in_features: int = 512,
        hidden_dim: Union[int, List[int]] = 32,
        out_features: int = 9,
        mixing_layer: str = "gp",
    ):
        super().__init__()

        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim]
        else:
            hidden_dims = list(hidden_dim)

        if len(hidden_dims) == 0:
            raise ValueError("hidden_dim must be a non-empty int or List[int]")

        self.mixing_layer = str(mixing_layer).lower()
        if self.mixing_layer not in {"gp", "mvlinear", "linear"}:
            raise ValueError("mixing_layer must be one of: gp, mvlinear, linear")

        self.blocks = nn.ModuleList()
        prev = in_features

        for hd in hidden_dims:
            self.blocks.append(
                nn.ModuleDict({
                    "fc": FullyConnectedSteerableGeometricProductLayer(
                        algebra, in_features=prev, out_features=hd
                    ),
                    "act1": MVSiLU(algebra, hd),
                    "mix": self._build_mixing_layer(algebra=algebra, channels=hd),
                    "act2": MVSiLU(algebra, hd),
                })
            )
            prev = hd

        self.out = FullyConnectedSteerableGeometricProductLayer(
            algebra, in_features=prev, out_features=out_features
        )

    def _build_mixing_layer(self, algebra, channels: int):
        if self.mixing_layer == "gp":
            return SteerableGeometricProductLayer(algebra, channels)
        if self.mixing_layer == "mvlinear":
            return MVLinear(algebra, channels, channels)
        if self.mixing_layer == "linear":
            return nn.Linear(channels * (2**algebra.dim), channels * (2**algebra.dim))
        raise ValueError(f"Unsupported mixing_layer: {self.mixing_layer}")

    def _apply_mixing(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.mixing_layer == "linear":
            b, n, d = x.shape
            x = x.reshape(b, n * d)
            x = layer(x)
            return x.reshape(b, n, d)
        return layer(x)

    def forward(self, x):
        if getattr(self, "_extra_tensors_device", None) != x.device:
            _move_unregistered_tensors_to_device(self, x.device)
            self._extra_tensors_device = x.device

        for b in self.blocks:
            x = b["fc"](x)
            x = b["act1"](x)
            x = self._apply_mixing(b["mix"], x)
            x = b["act2"](x)
        x = self.out(x)
        return x


class TralaleroCompetitor(nn.Module):
    def __init__(self, algebra, encoder_type: str = "resnet", ga_pool_hw: tuple = (28, 28)):
        super().__init__()
        self.algebra = algebra
        self._use_ga_backbone = encoder_type in {"ga", "ga_canonical"}
        self._mv_dim = int(2**algebra.dim)

        if self._use_ga_backbone:
            if len(ga_pool_hw) != 2:
                raise ValueError("ga_pool_hw must have exactly two elements: (height, width)")
            self.ga_pool_hw = (int(ga_pool_hw[0]), int(ga_pool_hw[1]))
            if self.ga_pool_hw[0] <= 0 or self.ga_pool_hw[1] <= 0:
                raise ValueError("ga_pool_hw values must be positive")

            self.pre_encode_pool = nn.AdaptiveAvgPool2d(self.ga_pool_hw)
            self.backbone = build_encoder(encoder_type)
            self._n_mv = int(self.ga_pool_hw[0] * self.ga_pool_hw[1])
        else:
            self._n_mv = 8
            self.backbone = build_encoder(encoder_type)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            enc_channels = getattr(self.backbone, "output_shape", None)[0]
            self.projective_matrix = nn.Linear(enc_channels, self._n_mv * self._mv_dim)

        self.ga_head = TralaleroTralala(algebra, in_features=self._n_mv)

    def _ga_to_canonical_mv(self, mv_grid: torch.Tensor) -> torch.Tensor:
        if mv_grid.shape[1] == self._mv_dim:
            return mv_grid

        if mv_grid.shape[1] != 6:
            raise ValueError(
                f"Unsupported GA encoder channels: expected 6 or {self._mv_dim}, got {mv_grid.shape[1]}"
            )

        e, e123, e1, e2, e13, e23 = torch.unbind(mv_grid, dim=1)
        zeros = torch.zeros_like(e)
        return torch.stack([e, e1, e2, zeros, zeros, e13, e23, e123], dim=1)


    def forward(self, x):
        if self._use_ga_backbone:
            pooled_x = self.pre_encode_pool(x)
            mv_grid = self.backbone(pooled_x)
            mv_grid = self._ga_to_canonical_mv(mv_grid)
            b, mv_dim, h, w = mv_grid.shape
            x = mv_grid.permute(0, 2, 3, 1).reshape(b, h * w, mv_dim)
        else:
            x = self.backbone(x)
            x = self.avgpool(x)
            x = x.flatten(1, -1)
            x = self.projective_matrix(x)
            x = x.reshape(x.shape[0], self._n_mv, self._mv_dim)

        x = self.ga_head(x)
        x = x[:, :, 0]
        x = x.reshape(x.shape[0], 3, 3)
        return x



class MLPBaseline(nn.Module):
    def __init__(self, encoder_type: str = "resnet"):
        super().__init__()
        self.backbone = build_encoder(encoder_type)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        enc_channels = getattr(self.backbone, "output_shape", None)[0]
        self.linear_head = nn.Linear(in_features=enc_channels, out_features=9)


    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.flatten(1, -1)
        x = self.linear_head(x)
        x = x.reshape(x.shape[0], 3, 3)
        return x


class I2S_Backbone(nn.Module):
    def __init__(
        self,
        algebra,
        lmax: int = 6,
        rec_level: int = 3,
        hidden_dim: List = [32],
        temperature: float = 1.0,
        backbone_name: str = "resnet50",
        pretrained_backbone: bool = True,
        freeze_backbone: bool = True,
        use_positional_encoding: bool = True,
        output_mode: str = "auto",
        mv_per_position: int = 1,
        adapter_mid_channels: int = 0,
        adapter_high_channels: int = 0,
        adapter_output_size: int = 16,
        ga_head_type: str = "tralalero",
        ga_head_mixing_layer: str = "gp",
    ):
        super().__init__()
        self.algebra = algebra
        self.lmax = int(lmax)
        self.rec_level = int(rec_level)
        self.temperature = float(temperature)
        self.backbone_name = str(backbone_name).lower()
        self.hidden_dim = hidden_dim

        self._mv_dim = int(2**algebra.dim)
        self.mv_per_position = int(mv_per_position)
        if self.mv_per_position <= 0:
            raise ValueError("mv_per_position must be positive")

        self.conv_adapter_output = int(adapter_output_size)
        if self.conv_adapter_output <= 0:
            raise ValueError("adapter_output_size must be positive")
        self._n_mv = self.conv_adapter_output ** 2 * self.mv_per_position
        if self._n_mv > 4096:
            print(
                f"Warning: I2S_Backbone uses {self._n_mv} multivector tokens "
                f"(adapter_output_size={self.conv_adapter_output}, "
                f"mv_per_position={self.mv_per_position}). "
                "This may be memory intensive."
            )
        adapter_out_channels = self.mv_per_position * self._mv_dim
        auto_mid_channels = max(64, adapter_out_channels * 2)
        auto_high_channels = max(256, auto_mid_channels * 2)

        self.adapter_mid_channels = (
            int(adapter_mid_channels)
            if int(adapter_mid_channels) > 0
            else auto_mid_channels
        )

        self.adapter_high_channels = (
            int(adapter_high_channels)
            if int(adapter_high_channels) > 0
            else auto_high_channels
        )

        if self.adapter_mid_channels < adapter_out_channels:
            raise ValueError(
                "adapter_mid_channels must be >= mv_per_position * mv_dim "
                f"({adapter_out_channels}), got {self.adapter_mid_channels}"
            )

        if self.adapter_high_channels < self.adapter_mid_channels:
            raise ValueError(
                "adapter_high_channels must be >= adapter_mid_channels, "
                f"got high={self.adapter_high_channels}, mid={self.adapter_mid_channels}"
            )

        if output_mode not in {"auto", "rotation_matrix", "fourier", "rotor", "multivector_rotor"}:
            raise ValueError("output_mode must be one of: auto, rotation_matrix, fourier, rotor, multivector_rotor")
        self.output_mode = output_mode
        if self.output_mode in {"rotor", "multivector_rotor"} and self._mv_dim != 8:
            raise ValueError(
                "I2S_Backbone rotor/multivector_rotor modes currently require Cl(3,0), "
                f"expected mv_dim=8, got mv_dim={self._mv_dim}."
            )

        self.backbone, backbone_channels = self._build_backbone(
            backbone_name=self.backbone_name,
            pretrained_backbone=pretrained_backbone,
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.conv_adapter = nn.Sequential(
            nn.Conv2d(backbone_channels, self.adapter_high_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.adapter_high_channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(
                self.adapter_high_channels,
                self.adapter_mid_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.adapter_mid_channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(
                self.adapter_mid_channels,
                adapter_out_channels,
                kernel_size=1,
                bias=True,
            ),

            nn.AdaptiveAvgPool2d((self.conv_adapter_output, self.conv_adapter_output)),
        )

        self.use_positional_encoding = bool(use_positional_encoding)
        if self.use_positional_encoding:
            self.positional_embedding = nn.Parameter(torch.zeros(1, self._n_mv, self._mv_dim))
            nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        self.ga_head_type = str(ga_head_type).lower()
        self.ga_head_mixing_layer = str(ga_head_mixing_layer).lower()

        self.num_coeffs = _so3_num_fourier_coeffs(self.lmax)
        self.ga_head_fourier = self._build_ga_head(out_features=self.num_coeffs)
        self.ga_head_rotation = self._build_ga_head(out_features=9)
        self.ga_head_rotor = self._build_ga_head(out_features=4)
        self.ga_head_mv_rotor = self._build_ga_head(out_features=1)

        xyx = so3_healpix_grid(rec_level=self.rec_level)
        wign = flat_wigner(self.lmax, *xyx)
        self.register_buffer("so3_xyx", xyx, persistent=False)
        self.register_buffer("so3_wigner_T", wign.transpose(0, 1).contiguous(), persistent=False)
        self.register_buffer("so3_rotmats_cache", o3.angles_to_matrix(*self.so3_xyx), persistent=False)

    def _build_ga_head(self, out_features: int):
        if self.ga_head_type == "tralalero":
            return TralaleroTralala(
                algebra=self.algebra,
                in_features=self._n_mv,
                hidden_dim=self.hidden_dim,
                out_features=out_features,
            )
        if self.ga_head_type == "transformer_like":
            return TransformerLikeMVHead(
                algebra=self.algebra,
                in_features=self._n_mv,
                hidden_dim=self.hidden_dim,
                out_features=out_features,
            )
        if self.ga_head_type == "reduced":
            return ReducedGeometricProductHead(
                algebra=self.algebra,
                in_features=self._n_mv,
                hidden_dim=self.hidden_dim,
                out_features=out_features,
                mixing_layer=self.ga_head_mixing_layer,
            )
        raise ValueError("Unsupported ga_head_type: " f"{self.ga_head_type}. Expected one of: tralalero, transformer_like, reduced")

    def _build_backbone(self, backbone_name: str, pretrained_backbone: bool):
        if backbone_name == "resnet50":
            backbone_weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained_backbone else None
            resnet = torchvision.models.resnet50(weights=backbone_weights)
            backbone = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
            )
            return backbone, 2048

        if backbone_name == "convnext_tiny":
            backbone_weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained_backbone else None
            convnext = torchvision.models.convnext_tiny(weights=backbone_weights)
            return convnext.features, 768

        raise ValueError(f"Unsupported backbone_name: {backbone_name}")

    def _resolve_mode(self):
        if self.output_mode != "auto":
            return self.output_mode
        return "fourier"

    def _encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.backbone(x)
        adapted = self.conv_adapter(fmap)

        b, c, h, w = adapted.shape

        expected_c = self.mv_per_position * self._mv_dim
        expected_h = self.conv_adapter_output
        expected_w = self.conv_adapter_output

        if (c, h, w) != (expected_c, expected_h, expected_w):
            raise RuntimeError(
                f"Expected adapted features [B, {expected_c}, {expected_h}, {expected_w}], "
                f"got [B, {c}, {h}, {w}]"
            )

        tokens = adapted.reshape(
            b,
            self.mv_per_position,
            self._mv_dim,
            h,
            w,
        )

        tokens = tokens.permute(0, 3, 4, 1, 2).reshape(
            b,
            h * w * self.mv_per_position,
            self._mv_dim,
        )

        if tokens.shape[1] != self._n_mv or tokens.shape[2] != self._mv_dim:
            raise RuntimeError(
                f"Expected tokens [B, {self._n_mv}, {self._mv_dim}], "
                f"got {list(tokens.shape)}"
            )

        if self.use_positional_encoding:
            tokens = tokens + self.positional_embedding

        return tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_extra_tensors_device", None) != x.device:
            _move_unregistered_tensors_to_device(self, x.device)
            self._extra_tensors_device = x.device

        mode = self._resolve_mode()
        tokens = self._encode_tokens(x)
        if mode == "fourier":
            coeffs_mv = self.ga_head_fourier(tokens)
            coeffs = coeffs_mv[..., 0]
            logits = self.logits_on_grid(coeffs)
            logits = logits / max(self.temperature, 1e-8)
            return logits
        if mode == "rotation_matrix":
            rot_mv = self.ga_head_rotation(tokens)
            rot = rot_mv[..., 0].reshape(x.shape[0], 3, 3)
            return rot
        if mode == "rotor":
            rotor_mv = self.ga_head_rotor(tokens)
            rotor = rotor_mv[..., 0]
            rotor = rotor / rotor.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            return rotor
        if mode == "multivector_rotor":
            mv = self.ga_head_mv_rotor(tokens)
            mv = mv[:, 0, :]
            return mv
        raise ValueError(f"Unsupported mode: {mode}")

    def logits_on_grid(self, coeffs: torch.Tensor) -> torch.Tensor:
        if coeffs.dim() == 3:
            coeffs = coeffs.squeeze(1)
        return torch.matmul(coeffs, self.so3_wigner_T)

    @torch.no_grad()
    def probs_on_grid(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict_rotmat(self, coeffs: torch.Tensor) -> torch.Tensor:
        probs = self.probs_on_grid(coeffs)
        idx = torch.argmax(probs, dim=-1)
        return idx

    @torch.no_grad()
    def predict(self, x):
        out = self.forward(x)
        mode = self._resolve_mode()

        if mode == "fourier":
            idx = self.predict_rotmat(out)
            return self.so3_rotmats_cache[idx]

        if mode == "rotation_matrix":
            return out

        if mode == "rotor":
            return _unit_quaternion_to_matrix(out)

        if mode == "multivector_rotor":
            return _mv_to_rotation_matrix(out)

        raise ValueError(f"Unsupported mode: {mode}")

    @torch.no_grad()
    def get_nearest_idx(self, rot_gt: torch.Tensor):
        return nearest_rotmat(rot_gt, self.so3_rotmats_cache)


I2S_ResNet = I2S_Backbone
