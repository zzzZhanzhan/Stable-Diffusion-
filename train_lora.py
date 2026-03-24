import os
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import FlowMatchEulerDiscreteScheduler

# 导入你已有的 LoRA 模块
from lora.sd3_lora import SD3LoRAModel, create_lora_config

# ====================== 训练配置（仅改这里）======================
BASE_MODEL_PATH = "./stable-diffusion-3-medium"  # 你的SD3路径
DATASET_DIR = "./your_dataset"                   # 图片+txt提示词，即 1.jpg 1.txt 2.jpg 2.txt
OUTPUT_DIR = "./sd3_lora_output"
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# LoRA 参数
LORA_RANK = 8
LORA_ALPHA = 16
# ======================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------- 数据集类（图片 + 提示词）----------------------
class ImageTextDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)   # 图像路径
        txt_path = os.path.splitext(img_path)[0] + ".txt"   # 文本路径
        with open(txt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, prompt

# ---------------------- 图像预处理 ----------------------
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# ---------------------- 加载数据集 ----------------------
dataset = ImageTextDataset(DATASET_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------------------- 初始化 SD3 LoRA 模型 ----------------------
lora_config = create_lora_config(rank=LORA_RANK, alpha=LORA_ALPHA, dropout=0.0)
model = SD3LoRAModel(
    base_model_path=BASE_MODEL_PATH,
    lora_config=lora_config,
    device=DEVICE,
    dtype=DTYPE
)

# Flow Matching 调度器
model.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
    model.pipe.scheduler.config,
    shift=3.0  # SD3 Medium
)
scheduler = model.pipe.scheduler

print(f"可训练 LoRA 参数: {model.get_trainable_parameters():,}")

# ---------------------- 优化器 ----------------------
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE,
    weight_decay=0.01
)

# ----------------------冻结整个基础模型，只训练 LoRA ----------------------
# 1. 冻结所有基础模型参数
model.transformer.requires_grad_(False)
model.vae.requires_grad_(False)
for encoders in model.text_encoders:
    if encoders is not None:
        encoders.requires_grad_(False)

# 2. 仅让 LoRA 层可训练
for layer in model.lora_layers.values():
    layer.requires_grad_(True)

# ---------------------- 训练循环（Flow Matching 版）----------------------
print("\n开始 SD3 Flow Matching LoRA 训练...\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for images, prompts in pbar:
        optimizer.zero_grad()
        images = images.to(DEVICE).to(DTYPE)

        with torch.no_grad():
            # pipe 集成的文本编码器
            prompt_embeds, pooled_embeds, _ = model.pipe.encode_prompt(
                prompts,
                device=DEVICE,
                do_classifier_free_guidance=False
            )

            # VAE 编码图像
            latents = model.vae.encode(images).latent_dist.sample()
            latents = latents * model.vae.config.scaling_factor


        # 3. Flow Matching 核心：随机 t ∈ [0,1]
        t = torch.rand(latents.shape[0], device=DEVICE).to(DTYPE)
        t = t.view(-1, 1, 1, 1)  # 广播到 latent 维度

        # 4. 采样先验 z ~ N(0,1)
        z = torch.randn_like(latents)

        # 5. 插值 x_t = (1−t)x₀ + tz
        x_t = (1 - t) * latents + t * z

        # 6. 真实速度 v_t = z - x₀
        v_t = z - latents

        # 7. MMDiT 预测速度 v_θ
        # SD3 调度器会把 t 转为内部 timestep
        timesteps = scheduler.scale_timestep(t * scheduler.config.num_train_timesteps)
        v_pred = model.transformer(
            hidden_states=x_t,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_embeds
        ).sample

        # 8. Flow Matching 损失：MSE(v_pred, v_t)
        loss = torch.nn.functional.mse_loss(v_pred, v_t)

        # 反向传播
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} | 平均损失: {avg_loss:.6f}")
    model.save_lora_weights(os.path.join(OUTPUT_DIR, f"sd3_lora_epoch_{epoch+1}.pth"))

model.save_lora_weights(os.path.join(OUTPUT_DIR, "sd3_lora_final.pth"))
print("\n训练完成！LoRA 已保存到:", OUTPUT_DIR)