import math
from urllib.parse import uses_relative

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torch.utils.checkpoint import checkpoint


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SwiGLUFeedForward(nn.Module):
    """
        门控前馈网络
    """
    def __init__(self, dim):
        super().__init__()

        hidden_dim = int( 8 * dim / 3)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class SigleDiTBlock(nn.Module):
    """
        MM-DiT 单分支
    """
    def __init__(self, hidden_dim):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)   # 第一个 LyaerNorm
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)   # 第二个 LyaerNorm

        # AdaLN
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )

        self.mlp = SwiGLUFeedForward(hidden_dim)

        self.linear = nn.Linear(hidden_dim, hidden_dim)

        # AdaLN 参数
        self.gate_msa = None

        self.shift_mlp = None
        self.scale_mlp = None
        self.gate_mlp = None

    def pre_attention(self, x, y):
        # 注意力之前 AdaLN
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(6, dim=1)
        self.gate_msa = gate_msa
        self.shift_mlp = shift_mlp
        self.scale_mlp = scale_mlp
        self.gate_mlp = gate_mlp
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        return x

    def post_attention(self, x, attn_out):
        # 注意力之后 AdaLN
        # x 表示原始输入， attn_out 为 attention 输出

        attn_out = self.gate_msa.unsqueeze(1) * self.linear(attn_out)  # AdaLN

        out = x + attn_out
        out = modulate(self.norm2(out), self.shift_mlp, self.scale_mlp)
        out = self.mlp(out)
        out = self.gate_mlp.unsqueeze(1) * out
        out = x + out
        return out


class MMDiTBlock(nn.Module):
    """
        MM-DiT 模型架构
    """
    def __init__(self, hidden_dim, num_heads, gradient_checkpoint, use_rmsnorm):
        super().__init__()

        assert hidden_dim // num_heads
        self.context_block = SigleDiTBlock(hidden_dim)   # 用于文本
        self.x_block = SigleDiTBlock(hidden_dim)         # 用于图像

        self.head_dim = hidden_dim // num_heads
        self.use_rmsnorm = use_rmsnorm
        if self.use_rmsnorm:
            self.rmsnorm = nn.RMSNorm(self.head_dim, elementwise_affine=False, eps=1e-6)

        self.qkv_c = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.qkv_x = nn.Linear(hidden_dim, 3 * hidden_dim)

        self.gradient_checkpoint = gradient_checkpoint

    def _forward(self, x, context, y):
        """
        x:       图像潜在空间
        context: 文本条件
        y:       时间步嵌入  AdaLN
        """
        B, L_context, D = context.shape
        B, L_x, _ = x.shape
        # 计算图像潜在空间 QKV
        x_modulate = self.x_block.pre_attention(x, y)
        qkv_x = self.qkv_x(x_modulate)
        q_x, k_x, v_x = qkv_x.reshape(B, L_x, -1, self.head_dim).chunk(3, dim=2)

        # 计算条件文本 QKV
        c_modulate = self.context_block.pre_attention(context, y)
        qkv_c = self.qkv_c(c_modulate)
        q_c, k_c, v_c = qkv_c.reshape(B, L_context, -1, self.head_dim).chunk(3, dim=2)

        if self.use_rmsnorm:
            q_c = self.rmsnorm(q_c)
            k_c = self.rmsnorm(k_c)
            q_x = self.rmsnorm(q_x)
            k_x = self.rmsnorm(k_x)

        q = torch.cat([q_c, q_x], dim=1).transpose(1, 2)
        k = torch.cat([k_c, k_x], dim=1).transpose(1, 2)
        v = torch.cat([v_c, v_x], dim=1).transpose(1, 2)

        # Cross Attention
        attn_out = F.scaled_dot_product_attention(q, k, v)    # 使用 PyTorch 优化的缩放点积
        attn_out = attn_out.transpose(1, 2).reshape(B, -1, D)

        c_attn_out = attn_out[:, :L_context]   # 条件
        x_attn_out = attn_out[:, L_context:]   # 图像

        c_out = self.context_block.post_attention(context, c_attn_out)
        x_out = self.x_block.post_attention(x, x_attn_out)

        return c_out, x_out

    def forward(self, x, context, y):
        # 使用梯度检查点保存（PyTorch）
        if self.training and self.gradient_checkpoint:
            return checkpoint(self._forward, x, context, y)
        else:
            return self._forward(x, context, y)


class PatchEmbed(nn.Module):
    """
        Patch Embedding 实现
        仅仅适用于固定尺寸
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        # 用卷积实现 Patch 切割
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias
        )

    def forward(self, x):
        # [B, C, H, W] -> [B, D, H/P, W/P]
        x = self.proj(x)

        # [B, D, H/P, W/P] -> [B, D, N] -> [B, N, D]
        x = x.flatten(2).transpose(1, 2)
        return x

class UnPatch(nn.Module):
    def __init__(self, hidden_dim, patch_size, out_channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.c = out_channels
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, patch_size**2 * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )

    def forward(self, x, con):
        b, n, _ = x.shape
        p = self.patch_size
        c = self.c
        w = h = int(n ** 0.5)
        shift, scale = self.adaLN_modulation(con).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)

        x = x.view(b, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(b, c, h * p, w * p)
        return x


class TimestepEmbedding(nn.Module):
    """
        时间步嵌入模块
    """
    def __init__(self, hidden_dim, freq_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_embedding_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.freq_embedding_size = freq_embedding_size

    def timestep_encoding(self, t, dim, max_period=10000):
        # 时间步嵌入
        half_dim = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim).to(
            t.device)
        t_freqs = t[:, None].float() * freqs[None, :]
        t_embedding = torch.cat([torch.cos(t_freqs), torch.sin(t_freqs)], dim=-1)
        return t_embedding

    def forward(self, t):
        t_embedding = self.timestep_encoding(t, self.freq_embedding_size)
        t_embedding = self.mlp(t_embedding)
        return t_embedding



class MMDiT(nn.Module):
    def __init__(self,
                 input_size,
                 patch_size,
                 in_channels,
                 hidden_dim,
                 con_dim,
                 num_heads,
                 num_blocks,
                 gradient_checkpoint,
                 use_rmsnorm):
        super().__init__()

        # x 转换为 token
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_dim, bias=True)
        num_patches = self.x_embedder.num_patches
        # 时间步嵌入
        self.t_embedder = TimestepEmbedding(hidden_dim)
        # 条件嵌入
        self.y_embedder = nn.Sequential(
                        nn.Linear(con_dim, hidden_dim),
                        nn.SiLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                         )

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim), requires_grad=False)
        self.use_checkpoint = gradient_checkpoint
        self.mmdit_blocks = nn.ModuleList(
            [
                MMDiTBlock(hidden_dim, num_heads, gradient_checkpoint, use_rmsnorm)
                for i in range(num_blocks)
            ]
            )

        self.final_layer = UnPatch(hidden_dim, patch_size, in_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # 初始化注意力 Linear:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # 位置编码嵌入
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # 初始化 Patch 嵌入
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # 初始化条件嵌入
        nn.init.normal_(self.y_embedder[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder[2].weight, std=0.02)

        # 初始化时间步嵌入
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.mmdit_blocks:
            nn.init.constant_(block.context_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.context_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.x_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.x_block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def forward(self, t, x, y):
        """
            x : [N, C, H, W]  图像的潜在表征
            t : 扩散模型的时间步
            y : 文本条件
        """
        B, C, H, W = x.shape

        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        y = self.y_embedder(y)

        for mmdit_block in self.mmdit_blocks:
            y, x = mmdit_block(x, y, t)   # x, y, t 对应于 x, context, y

        x = self.final_layer(x, t)    # [N, C, H, W]
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cla_token=False, extra_tokens=0):
    """
        2-D 位置编码
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_h, grid_w)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])

    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
    pos_emb = np.concatenate([emb_h, emb_w], axis=1)

    if cla_token and extra_tokens > 0:
        pos_emb = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_emb], axis=0)
    return pos_emb


def get_1d_sincos_pos_embed(embed_dim, pos):
    """
        1-D 位置编码
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1./ 10000*omega

    pos = pos.reshape(-1)
    out = pos[:, None] * omega[None, :]

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_cos, emb_sin], axis=1)  #
    return emb



if __name__ == "__main__":
    from thop import profile, clever_format
    config = dict(input_size = 32,
        patch_size = 4,
        in_channels = 3,
        hidden_dim = 1024,
        con_dim = 768,
        num_heads = 4,
        num_blocks = 3,
        gradient_checkpoint = False,
        use_rmsnorm = True)

    device = "cuda"
    sd3 = MMDiT(**config).to(device)

    x = torch.randn((10, 3, 32, 32)).to(device)
    t = torch.randn((10,)).to(device)
    y = torch.zeros((10, 77, 768)).to(device)

    # 分析计算量和参数量
    flops, params = profile(sd3, inputs=(t,x,y,))
    flops, params = clever_format([flops, params], "%.3f")

    print(f"FLOPs: {flops}")
    print(f"Parameters: {params}")











