import argparse
import os
import sys


import random
from collections import OrderedDict
from copy import deepcopy
import numpy as np
from torch.nn import functional as F
import torch
import torch.distributed as dist
from tqdm import tqdm

import clip
from datasets import load_dataset
from diffusers import AutoencoderKL
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
import logging
from torch.utils.data.distributed import DistributedSampler

from torchvision.transforms import transforms

from model.mini_diffusion import MMDiT


# ==================== 工具函数 ====================


def create_logger(logging_dir):
    """
    创建 logger 写入日志文件并进行标准处理
    """

    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)

    return logger

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU
    # 以下设置保证确定性，但可能降低性能
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


# ------------------- 1. 定义数据预处理变换 -------------------
# 图像预处理（可根据你的模型调整，比如ViT/ResNet的输入尺寸）
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 统一尺寸
    transforms.ToTensor(),  # 转Tensor: [H,W,C]→[C,H,W]，值归一化到0-1
    transforms.Normalize(  # 标准化（ImageNet均值/方差）
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class ImageTextDataset(Dataset):
    def __init__(self, data_path, image_transform=None):
        """
        初始化数据集
        Args:
            parquet_path: parquet文件路径
            image_transform: 图像预处理变换
            tokenizer: 文本tokenizer
        """
        # 读取parquet文件（按需用columns参数只加载需要的列，提升速度）
        dataset = load_dataset(
            "parquet",
            data_files=f"{data_path}*.parquet",  # 使用通配符匹配所有parquet文件
        )
        self.dataset = dataset["train"]
        self.transform = image_transform

    def __len__(self):
        """返回数据集总长度"""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        加载单条数据（核心方法）
        Returns:
            dict: 包含预处理后的图像、文本输入等
        """
        # 1. 读取单条数据
        data = self.dataset[idx]

        # 2. 处理图像
        img = data["image"]  # 二进制列表（对应多张图像）

        # 图像预处理
        if self.transform:
            img = self.transform(img)

        # 3. 处理文本（alt_text是嵌套列表，先展平）
        text = data["alt_text"][0].strip()  # 格式：[['xxx']]
        # 4. 返回训练所需数据（按需添加label等字段）
        return img,  text


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()



def main(model_config):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    device = "cuda:1"
    seed = 42
    set_seed(seed)

    image_size = 256
    # 设置主进程:
    results_dir = "/media/gpu-1/BOX/z/mysd/result_dir"
    vae_path = "/media/gpu-1/BOX/z/mysd/sd-vae-ft-mse"
    clip_path = "/media/gpu-1/BOX/z/mysd/clip"
    data_path = "/media/gpu-1/BOX/z/mysd/data/data_raw/"
    batch_size = 16
    epochs = 10

    os.makedirs(results_dir, exist_ok=True)        # 存储结果路径
    checkpoint_dir = f"{results_dir}/checkpoints"  # 存储检查点
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(results_dir)
    logger.info(f"Experiment directory created at {results_dir}")


    # 创建模型大小:
    assert image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."

    model = MMDiT(**model_config).to(device)
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    # 加载 vae
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    clip_model, preprocess = clip.load("ViT-L/14", device=device, download_root=clip_path)  # 指定下载目录
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0)

    # Setup data:
    # 初始化Dataset
    dataset = ImageTextDataset(
        data_path=data_path,
        image_transform=image_transform,
    )
    # 构建DataLoader（添加pin_memory=True加速GPU加载）
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,  # 锁页内存，提升数据传输速度
        drop_last=True    # 丢弃最后一个不完整批次（可选）
    )


    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0

    logger.info(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for image, text in tqdm(dataloader, desc=f"当前 epochs = {epoch}"):
            image = image.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                latents = vae.encode(image).latent_dist.sample().mul_(0.18215) # [N, 4, 32, 32]

                # text encoder 也可以考虑最后一层的特征作为嵌入
                cond_text_tokens = clip.tokenize(text, truncate=True,).to(device)
                cond_text_embed= clip_model.token_embedding(cond_text_tokens).to(device)
                cond_text_embed = cond_text_embed.to(torch.float32)

                # 准备文本嵌入（替换无条件为空）
                # 无条件文本嵌入（空文本，CFG的对比分支）
                uncond_text_embed = torch.zeros_like(cond_text_embed)
                # CFG训练核心：随机替换10%样本为无条件文本（
                # 生成掩码：10%概率为True（替换为空文本）
                uncond_mask = torch.rand(batch_size, device=device) < 0.1
                # 扩展掩码维度，匹配文本嵌入形状（避免维度不匹配）
                uncond_mask_expand = uncond_mask.unsqueeze(-1).unsqueeze(-1).expand(cond_text_embed.shape)

                # 构建最终文本嵌入（条件/无条件混合）
                # 逻辑：uncond_mask为True时用空文本嵌入，否则用条件文本嵌入
                text_embed = torch.where(
                    uncond_mask_expand,  # 掩码：(B, D)，D为文本嵌入维度
                    uncond_text_embed,  # 空文本嵌入（无条件分支）
                    cond_text_embed  # 输入文本嵌入（条件分支）
                )


            # 采用 Flow Matching
            #  噪声latents（Flow Matching的目标分布，原代码是nose_latents笔误）
            noise_latents = torch.randn_like(latents).to(device)  # Flow Matching通常用标准正态分布作为目标

            # 采样时间步t（Flow Matching核心，t∈[0,1]）
            t = torch.rand(batch_size, dtype=torch.float).to(device)

            # 构建Flow Matching的插值分布 x_t = (1-t)*latents + t*noise_latents
            latents_t = (1 - t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) * latents + t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * noise_latents

            # 3计算真实速度场 v_true = noise_latents - latents（Flow Matching的目标）
            v_true = noise_latents - latents

            # 4.3 预测速度场v_pred（Flow Matching模型输出）
            # Flow Matching模型输入：x_t (latent) + t (时间步) + text_embed (文本嵌入)
            v_pred = model(t, latents_t, text_embed)  # 假设模型接收这三个输入

            # 损失函数
            loss = F.mse_loss(v_pred, v_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % 1000 == 0:
                # Measure training speed:
                torch.cuda.synchronize()

                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0

            # Save DiT checkpoint:
            if train_steps % 1000 == 0 and train_steps > 0:
                    checkpoint = {
                        "model": model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")


    logger.info("Done!")



if __name__ == "__main__":
    import torch.multiprocessing as mp
    model_config = dict(input_size=32,
                  patch_size=4,
                  in_channels=4,
                  hidden_dim=1024,
                  con_dim=768,
                  num_heads=4,
                  num_blocks=3,
                  gradient_checkpoint=False,
                  use_rmsnorm=True)
    main(model_config)

