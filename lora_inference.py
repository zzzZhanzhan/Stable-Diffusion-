import torch
from lora.sd3_lora  import SD3LoRAModel, create_lora_config

# ====================== 配置 ======================
BASE_MODEL_PATH = "./stable-diffusion-3-medium"  # "./sd3-medium" 路径
LORA_PATH = "./sd3_lora_output" #   训练好的LoRA权重
DEVICE = "cuda"
DTYPE = torch.bfloat16
# ==================================================

# 创建 LoRA 模型
lora_config = create_lora_config(rank=8, alpha=16)
model = SD3LoRAModel(
    base_model_path=BASE_MODEL_PATH,
    lora_config=lora_config,
    device=DEVICE,
    dtype=DTYPE
)

# 加载训练好的 LoRA 权重
model.load_lora_weights(LORA_PATH)
# 合并 LoRA 权重
model.merge_lora_weights()

# 3. 切换到推理模式
model.eval()

# ====================== 开始推理生成图片 ======================
prompt = "prompts"
negative_prompt = "blurry, low quality, ugly"    # 负提示词

# 使用 pipe 推理，仅替换 LoRA 权重
with torch.no_grad():
    image = model.pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=28,
        guidance_scale=7.0,
        output_type="pil"
    ).images[0]

# 保存
image.save("sd3_lora_infer.png")
print("✅ 生成完成！")