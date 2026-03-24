## Stable Diffusion 3 模型架构复现

### 目录结构

```
sd3/
├── clip                          # 存放 clip 模型权重
├── sd-vae-ft-mse                 # 存放 vae 模型权重  
├── data
     └── data_filter.py           # 数据清洗
├── model
     └── mini_diffusion.py        # SD3 架构实现（Flow Matching）
├── lora
     └── sd3_lora.py              # lora 组件   
├── evaluate
     └── clip_scores.py           # CLIP Score 计算 
├── train_mini_diffusion.py       # 简易 SD3 训练框架  
├── mini_diffusion_inference.py   # 简易 SD3 推理框架  
├── train_lora.py                 # 简易 SD3 lora 训练框架  
├── lora_inference.py             # 简易 SD3 lora 推理框架 
```



### 环境配置

使用 `diffusers` 来方便加载 SD 模型组件。

### 数据清洗

数据清洗过程包括 去除损坏的图像，文本去重，MiniHash 去重，图像感知哈希，CLIP相似度去重。

```python
python data_filter.py
```



### MM-DiT 架构

旨在实现简单的 MM-DiT 架构，模型大小也仅限于在 RTX 2080 Ti 上实现，可根据实际显存修改模型大小。

具体模型架构可参考：

[Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/pdf/2403.03206)

模型训练采用 flow matching ，可运行 `python train_mini_diffusion.py `

### LoRA 微调

数据集满足 图片+同名txt提示词

```bash
your_dataset/
├── img01.jpg
├── img01.txt  # 
├── img02.png
└── img02.txt  # 
......
```



LoRA 参数配置

```python
"""LoRA 配置参数"""
r: int = 8                          # LoRA 秩
lora_alpha: int = 16                # 缩放参数
target_modules: List[str] = None    # 目标模块列表
lora_dropout: float = 0.0           # Dropout 概率
bias: str = "none"                  # 偏置训练模式
```

运行训练脚本 `python train_lora.py`



### 模型评估

使用 CLIP Score 来计算图文相似度来进行模型生成效果评估。运行 `python evaluate/clip_scores.py`

对于 FiD 分数，可以使用 `pytorch_fid` 或者 `torch-fidelity` 库来实现。
