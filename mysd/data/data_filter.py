import gc
import hashlib
import os
import random
from typing import List, Dict, Set

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from clip import clip
from datasets import load_dataset
from tqdm import tqdm

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from io import BytesIO
from PIL import Image


def filter_by_image_quality(data, min_size: int = 224) -> List[Dict]:
    """筛选低质量图像（过小或损坏）"""
    quality_data = []

    # 关键优化1：分批处理Dataset，避免一次性加载所有数据到内存
    batch_size = 2000  # 根据你的CPU/内存调整，内存小就调小（如1000）
    total_samples = len(data)
    total_batches = (total_samples + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(total_batches), desc="图像质量筛选"):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, total_samples)

        batch_data = data.select(range(start, end))
        for item in batch_data:
            image = item['image']

            # 检查图像尺寸
            if image.width < min_size or image.height < min_size:
                continue

            # 检查图像是否损坏（尝试转换为numpy数组）
            try:
                img_array = np.array(image)
                if img_array.size == 0:
                    continue
            except:
                continue

            quality_data.append(item)

            # 关键优化2：手动释放批次内存，触发垃圾回收
        del batch_data
        gc.collect()  # 强制清理未使用的内存，降低内存占用


    print(f"图像质量筛选: {len(data)} → {len(quality_data)} (去除 {len(data) - len(quality_data)} 条)")
    return quality_data


def exact_text_deduplication(data) -> List[Dict]:
    """基于MD5的精确文本去重"""
    seen_hashes = set()     # 存放唯一的哈希值
    unique_data = []        # 返回唯一的数据

    for item in tqdm(data, desc="文本精确去重"):
        text = item['alt_text'][0].strip()                        # 针对于 AnyModal/flickr30k 数据集
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()

        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            unique_data.append(item)

    print(f"精确去重: {len(data)} → {len(unique_data)} (去除 {len(data) - len(unique_data)} 条)")
    return unique_data


# ---------------------- MinHash核心工具 ----------------------
def generate_hash_seeds(k: int = 128) -> List[int]:
    """生成k个随机哈希种子（k越大越准，内存越大）"""
    random.seed(42)  # 固定种子保证可复现
    return [random.randint(0, 10**9) for _ in range(k)]

def text_to_shingles(text: str, n: int = 2) -> Set[str]:
    """文本转n-gram shingle集合（避免单字敏感）"""
    text = text.strip().lower()
    return {text[i:i+n] for i in range(len(text)-n+1)} if len(text)>=n else {text}

def minhash_signature(s: Set[str], seeds: List[int]) -> List[int]:
    """生成MinHash签名"""
    return [min(int(hashlib.md5((str(seed)+elem).encode()).hexdigest(),16) for elem in s) for seed in seeds]

# ---------------------- MinHash去重 ----------------------
def text_minhash_deduplication(data, threshold: float = 0.8, k: int = 128, batch_size: int = 2000):
    """
    MinHash近似去重（适配Dataset，低内存）
    :param threshold: 相似度阈值（>则去重）
    :param k: 签名维度（128常用）
    """
    seeds = generate_hash_seeds(k)
    seen_signatures = set()  # 存已出现的签名（元组可哈希）
    unique_data = []
    total_samples = len(data)
    total_batches = (total_samples + batch_size -1) // batch_size

    for batch_idx in tqdm(range(total_batches), desc="MinHash近似去重"):
        start = batch_idx * batch_size
        end = min((batch_idx+1)*batch_size, total_samples)
        batch_dataset = data[start:end]  # 原生Dataset，低内存

        for item in batch_dataset:
            try:
                # 1. 提取文本并转shingle
                alt_text = item.get('alt_text', [])
                text = alt_text[0].strip() if isinstance(alt_text, list) and len(alt_text)>0 else ""
                if not text:
                    continue
                shingles = text_to_shingles(text, n=2)

                # 2. 生成MinHash签名
                sig = minhash_signature(shingles, seeds)
                sig_tuple = tuple(sig)  # 转元组存入集合

                # 3. 去重判断（近似）
                if sig_tuple not in seen_signatures:
                    seen_signatures.add(sig_tuple)
                    unique_data.append(item)

            except Exception as e:
                print(f"跳过异常数据: {e}")
                continue

        del batch_dataset
        gc.collect()  # 强制回收内存

    print(f"MinHash去重: {total_samples} → {len(unique_data)} (去除 {total_samples-len(unique_data)} 条)")
    return unique_data


def _binary_to_hex(binary_list):
    """将二进制列表转换为16进制字符串"""
    # 每4位二进制转1位16进制
    hex_str = ""
    for i in range(0, len(binary_list), 4):
        chunk = binary_list[i:i + 4]
        chunk_str = ''.join(map(str, chunk))
        hex_char = hex(int(chunk_str, 2))[2:]  # 去掉0x前缀
        hex_str += hex_char
    return hex_str


def compute_p_hash(image_input, hash_size=32, dct_size=8):
    """
    计算图像的感知哈希（pHash）
    :param image_path: 图像文件路径
    :param hash_size: 缩放后的尺寸（建议32，需≥dct_size）
    :param dct_size: DCT变换后保留的低频区域尺寸（建议8）
    :return: 64位哈希字符串（16进制）、原始二进制哈希列表
    """
    try:
        # 1. 读取图像并转为灰度图
        if isinstance(image_input, str):
            # 输入是路径：按原逻辑读取
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"无法读取图像：{image_input}")
        elif isinstance(image_input, Image.Image):
            # 输入是PIL图像对象：转为OpenCV格式（numpy数组）
            # 1. PIL → numpy数组（RGB格式）
            img = np.array(image_input)
            # 2. RGB → BGR（OpenCV默认通道顺序）
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 核心：RGB→BGR
        else:
            raise TypeError(f"不支持的输入类型：{type(image_input)}，仅支持路径字符串或PIL图像对象")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 2. 缩放至固定尺寸（消除尺寸影响）
        resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)

        # 3. 转换为浮点型，避免DCT计算溢出
        resized_float = np.float32(resized)

        # 4. 离散余弦变换（DCT）：提取低频信息（核心步骤）
        dct = cv2.dct(resized_float)

        # 5. 保留DCT左上角的低频区域（忽略高频噪声）
        dct_low = dct[:dct_size, :dct_size]

        # 6. 计算低频区域的均值（排除第一个像素，它是直流分量）
        mean = np.mean(dct_low[1:])  # 跳过DC分量，提升鲁棒性

        # 7. 二值化：大于均值记1，否则记0（生成二进制哈希）
        binary_hash = (dct_low > mean).flatten().astype(int).tolist()

        # 8. 转换为16进制哈希字符串（方便存储和展示）
        hex_hash = _binary_to_hex(binary_hash)

        return hex_hash

    except Exception as e:
        print(f"计算pHash出错：{e}")
        return None

def hamming_distance(hash1: str, hash2: str) -> int:
    """计算汉明距离"""
    if hash1 is None or hash2 is None:
        return 999
    try:
        x = int(hash1, 16) ^ int(hash2, 16)
        return bin(x).count('1')
    except:
        return 999

def image_hash_deduplication(data, threshold: int = 10):
    """ 基于pHash的图像去噪"""
    print("计算图像感知哈希")
    hashes = {}

    # 根据图像 ID 去重，避免重复计算相同图像的 hash
    unique_images = {}
    for item in data:
        img_id = item['img_id']
        if img_id not in unique_images:
            unique_images[img_id] = item['image']

    # 计算图像 pHash 哈希
    # 计算每个唯一图像的pHash
    for img_id, image in tqdm(unique_images.items(), desc="计算图像哈希"):
        phash = compute_p_hash(image)
        if phash:
            hashes[img_id] = phash

    # 聚类相似图像
    print("聚类相似图像...")
    visited = set()
    unique_groups = []

    for img_id, h in tqdm(hashes.items(), desc="图像聚类"):
        if img_id in visited:
            continue

        # 找到所有相似图像
        similar_group = [img_id]
        visited.add(img_id)

        # 时间复杂度 O(n^2) 后续有时间再修改
        for other_id, other_h in hashes.items():
            if other_id not in visited and hamming_distance(h, other_h) < threshold:
                similar_group.append(other_id)
                visited.add(other_id)

        # 保留每组第一个
        unique_groups.append(similar_group[0])

    # 过滤数据：只保留唯一图像组中的图像
    unique_set = set(unique_groups)
    unique_data = [item for item in data if item['img_id'] in unique_set]

    print(f"图像去重: {len(data)} → {len(unique_data)} (去除 {len(data) - len(unique_data)} 条)")
    return unique_data




def compute_similarity(image_input, text_input, model_path, device):
    """
    计算单张图像和单段文本的相似度
    :param image_input: 支持3种输入：
                       1. PIL图像对象（如PIL.JpegImageFile）
                       2. 图像文件路径（str）
                       3. 图像numpy数组（需兼容PIL读取）
    :param text_input: 文本字符串（如"一只黑色的猫"）
    :return: 余弦相似度（0~1之间，值越高越匹配）
    """
    try:
        # 1. 预处理图像（转为模型可接受的格式）

        model, preprocess = clip.load("ViT-L/14",
                                      device="cuda",
                                      download_root=model_path)  # 指定下载目录

        if isinstance(image_input, str):
            # 输入是路径：读取为PIL对象
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            # 输入是PIL对象：直接使用
            image = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            # 输入是numpy数组：转为PIL对象
            image = Image.fromarray(image_input).convert("RGB")
        else:
            raise TypeError(f"不支持的图像输入类型：{type(image_input)}")

        # 图像预处理（仅图像！）
        image_tensor = preprocess(image).unsqueeze(0).to(device)  # 增加batch维度

        # ========== 3. 处理文本（CLIP原生文本编码） ==========
        # 文本tokenize：clip.tokenize 是原生文本处理函数
        text_tensor = clip.tokenize([text_input]).to(device)  # 转成token并加batch维度

        # ========== 4. 模型推理提取特征 ==========
        with torch.no_grad():  # 禁用梯度，提升速度
            # 提取图像和文本特征
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_tensor)

        # ========== 5. 计算余弦相似度 ==========
        # 归一化特征（CLIP官方要求）
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # 计算点积（即余弦相似度）
        similarity = (image_features @ text_features.T).item()

        return round(similarity, 4)  # 保留4位小数，便于查看

    except Exception as e:
        print(f"计算相似度出错：{e}")
        return 0.0


def filter_by_similarity(data: list[dict], threshold: float = 0.25, model_path="/media/gpu-1/BOX/z/mysd/clip"):
    """
    批量过滤图文对：保留相似度≥阈值的内容
    :param data: 输入数据，格式为字典列表，每个字典包含：
                 {"img_id": "xxx", "image": PIL对象/路径, "text": "描述文本"}
    :param threshold: 相似度阈值（经验值：0.25~0.35，可根据业务调整）
    :return: 过滤后的字典列表
    """
    filtered_data = []
    for item in tqdm(data, desc="图像文本相似度计算"):
        image = item["image"]
        text = item["alt_text"][0]

        # 计算相似度
        similarity = compute_similarity(image, text, model_path, "cuda")
        # 判断是否保留
        if similarity >= threshold:
            item["similarity"] = similarity  # 新增相似度字段，便于后续分析
            filtered_data.append(item)

    print(f"\n过滤完成：原始{len(data)}条 → 保留{len(filtered_data)}条")
    return filtered_data


def save_filtered_data(filtered_data_list, output_path):
    """
    保存过滤后的数据，仅保留image和alt_text字段

    Args:
        filtered_data_list (list): 过滤后的1000条数据列表，每条数据是字典格式
        output_path (str): 输出parquet文件路径，如 "filtered_data.parquet"
    """
    # 准备要保存的数据
    processed_data = []
    for item in tqdm(filtered_data_list, desc="保存过滤后的图像"):
        # 1. 处理图像：将PIL Image转换为字节流
        img = item['image']
        img_byte_arr = BytesIO()
        # 保存为JPEG格式（可根据需要改为PNG）
        img.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()

        # 2. 提取alt_text
        alt_text = item['alt_text'][0]

        # 3. 构建新的字典
        processed_data.append({
            'image': img_bytes,
            'text': alt_text
        })

    # 转换为pandas DataFrame
    df = pd.DataFrame(processed_data)

    # 转换为pyarrow Table并保存（确保图像字节流正确存储）
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path)
    output_path = os.path.join(output_path, 'filtered_data.parquet')
    print(f"数据已成功保存到 {output_path}，共 {len(filtered_data_list)} 条记录")


if __name__ == "__main__":
    # 文件夹路径（指向包含所有parquet文件的目录）
    folder_path = "/media/gpu-1/BOX/z/mysd/data/data_raw/"

    # 读取文件夹下所有parquet文件
    dataset = load_dataset(
        "parquet",
        data_files=f"{folder_path}*.parquet",  # 使用通配符匹配所有parquet文件
    )
    dataset = dataset["train"]
    print(f"原始数据共有 {len(dataset)} 图文数据对")
    print(dataset[1])
    print(dataset[0])

    # 去除损坏的图像文件
    # data = filter_by_image_quality(dataset.select(range(0, 10000)), 256)

    # dataset = dataset[0:10]
    # 执行基于 MD5 的文本去重
    # data = exact_text_deduplication(data)

    # 执行基于 MiniHash 进行去重
    # data = text_minhash_deduplication(data)

    # 执行图像感知哈希
    # data = image_hash_deduplication(data)

    # 执行 CLIP 相似度过滤 （需要时间较长）
    # data = filter_by_similarity(data, 0.5, model_path="/media/gpu-1/BOX/z/mysd/clip")

    # 保存清晰后的数据
    # save_filtered_data(data, output_path="/media/gpu-1/BOX/z/mysd/data/data_filter")

