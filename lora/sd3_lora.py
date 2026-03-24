"""
Stable Diffusion 3 LoRA 模块实现
基于 diffusers 库和 PEFT 方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class LoRAConfig:
    """LoRA 配置参数"""
    r: int = 8                          # LoRA 秩
    lora_alpha: int = 16                # 缩放参数
    target_modules: List[str] = None    # 目标模块列表
    lora_dropout: float = 0.0           # Dropout 概率
    bias: str = "none"                  # 偏置训练模式

    def __post_init__(self):
        if self.target_modules is None:
            # SD3 默认目标模块
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "out_proj",  # Attention
                # "fc1", "fc2",                              # MLP
                # "proj_in", "proj_out"                      # Projection
            ]


class LoRALayer(nn.Module):
    """
    标准 LoRA 层实现
    使用低秩分解来微调预训练权重: W = W_0 + BA
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # 低秩矩阵 A 和 B
        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))

        # Dropout
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # 初始化: A 使用 Kaiming 均匀初始化, B 使用零初始化
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播: 计算 BA * x * scaling"""
        x = self.dropout(x)
        return F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling


class LinearWithLoRA(nn.Module):
    """
    包装线性层，添加 LoRA 旁路
    """
    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.lora = LoRALayer(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        # 冻结原始权重
        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始输出 + LoRA 输出
        original_out = self.original_layer(x)
        lora_out = self.lora(x)
        return original_out + lora_out


class SD3LoRAModel(nn.Module):
    """
    Stable Diffusion 3 的 LoRA 模型管理器
    支持 MMDiT 和文本编码器的 LoRA 注入
    """

    def __init__(
        self,
        base_model_path: str,
        lora_config: LoRAConfig,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.lora_config = lora_config
        self.device = device
        self.dtype = dtype

        # 加载基础模型 (使用 diffusers)
        self._load_base_model(base_model_path)

        # 注入 LoRA 层
        self.lora_layers = nn.ModuleDict()
        self._inject_lora_to_transformer()

    def _load_base_model(self, model_path: str):
        """加载 SD3 基础模型"""
        from diffusers import StableDiffusion3Pipeline

        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            # text_encoder_3=None,       # T5 编码器
            # tokenizer_3=None,
        )
        self.pipe = self.pipe.to(self.device)

        # 提取核心组件
        self.transformer = self.pipe.transformer     # 模型主架构 MM-DiT 架构
        self.vae = self.pipe.vae
        self.text_encoders = [
            self.pipe.text_encoder,      # CLIP-L
            self.pipe.text_encoder_2,    # CLIP-G
            self.pipe.text_encoder_3,    # T5
        ]

    def _inject_lora_to_transformer(self):
        """向 MMDiT Transformer 注入 LoRA 层"""
        transformer = self.transformer
        config = self.lora_config

        # 遍历所有需要添加 LoRA 的模块
        for name, module in transformer.named_modules():
            if isinstance(module, nn.Linear):
                # 检查是否是目标模块
                if any(target in name for target in config.target_modules):

                    parent_name = ".".join(name.split(".")[:-1])
                    child_name = name.split(".")[-1]


                    parent = transformer
                    for part in parent_name.split("."):
                        if part:
                            parent = getattr(parent, part)

                    # 替换为 LoRA 包装层
                    lora_layer = LinearWithLoRA(
                        module,
                        r=config.r,
                        lora_alpha=config.lora_alpha,
                        lora_dropout=config.lora_dropout,
                    )
                    setattr(parent, child_name, lora_layer)
                    self.lora_layers[name] = lora_layer

        # print(f"✅ 成功注入 {len(self.lora_layers)} 个 LoRA 层")

    def get_trainable_parameters(self) -> int:
        """获取可训练参数数量"""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total

    def save_lora_weights(self, save_path: str):
        """保存 LoRA 权重"""
        lora_state_dict = {}

        # 只保存 LoRA 参数
        for name, layer in self.lora_layers.items():
            lora_state_dict[f"{name}.lora_A"] = layer.lora.lora_A.data
            lora_state_dict[f"{name}.lora_B"] = layer.lora.lora_B.data

        # 添加配置信息
        lora_state_dict["config"] = {
            "r": self.lora_config.r,
            "lora_alpha": self.lora_config.lora_alpha,
            "target_modules": self.lora_config.target_modules,
        }

        torch.save(lora_state_dict, save_path)
        print(f"💾 LoRA 权重已保存至: {save_path}")

    def load_lora_weights(self, load_path: str):
        """加载 LoRA 权重"""
        state_dict = torch.load(load_path, map_location=self.device)

        # 加载权重
        for name, layer in self.lora_layers.items():
            if f"{name}.lora_A" in state_dict:
                layer.lora.lora_A.data = state_dict[f"{name}.lora_A"]
                layer.lora.lora_B.data = state_dict[f"{name}.lora_B"]

        print(f"📂 LoRA 权重已从 {load_path} 加载")

    def merge_lora_weights(self):
        """合并 LoRA 权重到基础模型（用于推理加速）"""
        for name, layer in self.lora_layers.items():
            # 计算合并后的权重: W_merged = W_original + B @ A * scaling
            merged_weight = (
                layer.original_layer.weight.data +
                (layer.lora.lora_B @ layer.lora.lora_A) * layer.lora.scaling
            )
            layer.original_layer.weight.data = merged_weight

        print("🔀 LoRA 权重已合并到基础模型")

    def forward(self, *args, **kwargs):
        """前向传播委托给 transformer"""
        return self.transformer(*args, **kwargs)


# 创建 LoRA 配置
def create_lora_config(
    rank: int = 8,
    alpha: Optional[int] = None,
    target_modules: Optional[List[str]] = None,
    dropout: float = 0.0,
) -> LoRAConfig:
    """创建 LoRA 配置的便捷函数"""
    if alpha is None:
        alpha = rank * 2
    return LoRAConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
    )



