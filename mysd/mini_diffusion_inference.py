import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL
from tqdm import tqdm

import clip
from model.mini_diffusion import MMDiT


@torch.no_grad()
def sample(model, text_embed, num_steps, cfg_scale, device):
    """
        使用欧拉法解 ODE: dz/dt = v_theta(z, t, c)
    """
    # 从噪声开始
    z = torch.randn(1, 4, 32, 32, device=device)

    # 3. 构建空文本嵌入（无条件）
    empty_text_embed = torch.zeros_like(text_embed, device=device)

    # 时间步
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)  # 1 -> 0

    for i in tqdm(range(num_steps), desc="Sampling"):
        t = timesteps[i].unsqueeze(0)
        dt = timesteps[i + 1] - timesteps[i]  # 负值

        # 预测速度
        v_con = model(t, z, text_embed)   # 条件
        v_uncon = model(t, z, empty_text_embed)  # 无条件

        v_guide = v_uncon + (v_con - v_uncon) * cfg_scale

        # 欧拉步进
        z = z + v_guide * dt.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


    return z


if __name__ == "__main__":
    vae_path = "/media/gpu-1/BOX/z/mysd/sd-vae-ft-mse"
    clip_path = "/media/gpu-1/BOX/z/mysd/clip"
    checkponit_path = "/media/gpu-1/BOX/z/mysd/result_dir/checkpoints/0018000.pt"

    device = "cuda:1"
    clip_model, preprocess = clip.load("ViT-L/14", device=device, download_root=clip_path)  # 指定下载目录
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)

    model_config = dict(input_size=32,
                  patch_size=4,
                  in_channels=4,
                  hidden_dim=1024,
                  con_dim=768,
                  num_heads=4,
                  num_blocks=3,
                  gradient_checkpoint=False,
                  use_rmsnorm=True)
    model = MMDiT(**model_config).to(device)

    prompt = "a dog in the desk"
    cond_text_tokens = clip.tokenize(prompt, truncate=True, ).to(device)
    cond_text_embed = clip_model.token_embedding(cond_text_tokens).to(device)
    cond_text_embed = cond_text_embed.to(torch.float32)

    checkpoint = torch.load(checkponit_path, map_location=device)

    state_dict = checkpoint["ema"]
    model.load_state_dict(state_dict, strict=True)

    model.eval()

    z = sample(model, cond_text_embed, 100, 1.5, device)


    with torch.no_grad():
        z = z * 0.18215
        decoded_image = vae.decode(z)
        image = decoded_image.sample

    image = image.float()
    image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
    decoded_np = 255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)
    decoded_np = decoded_np.astype(np.uint8)
    out_image = Image.fromarray(decoded_np)

    out_image.show()



