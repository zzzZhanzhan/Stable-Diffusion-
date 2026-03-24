import torch
import clip
from PIL import Image
from typing import List

# ----------------------
# CLIP Score
# ----------------------
class CLIPScoreEvaluator:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-L/14", device=device)
        self.model.eval()

    @torch.no_grad()
    def compute_score(self, image: Image.Image, text: str) -> float:
        """
        计算单张图片与单文本的 CLIP Score
        """
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        text = clip.tokenize([text]).to(self.device)

        # 提取特征
        img_feat = self.model.encode_image(image)
        text_feat = self.model.encode_text(text)

        # 归一化
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        # 计算相似度
        similarity = torch.cosine_similarity(img_feat, text_feat, dim=-1).item()
        return float(similarity)

    @torch.no_grad()
    def compute_batch_score(self, images: List[Image.Image], texts: List[str]) -> float:
        """
        批量计算 CLIP Score（更快）
        """
        assert len(images) == len(texts)

        image_tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        text_tokens = clip.tokenize(texts).to(self.device)

        img_feat = self.model.encode_image(image_tensors)
        text_feat = self.model.encode_text(text_tokens)

        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        similarities = torch.cosine_similarity(img_feat, text_feat, dim=-1)
        return float(similarities.mean().item())


def evaluate_with_clip(
    image_path: str,
    prompt: str,
    device: str = "cuda"
):
    evaluator = CLIPScoreEvaluator(device=device)
    image = Image.open(image_path).convert("RGB")
    score = evaluator.compute_score(image, prompt)
    return score

if __name__ == "__main__":

    image_path = ""
    prompt = ""

    score = evaluate_with_clip(image_path, prompt)
    print(f"✅ CLIP Score = {score:.4f}")