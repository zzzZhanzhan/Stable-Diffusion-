

import torch



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================================== #
# from pytorch_fid import fid_score
# # 真实图像文件夹 vs 生成图像文件夹
# real_images_folder = "path/to/real_images"
# generated_images_folder = "path/to/generated_images"
#
# # 计算 FID
# fid_value = fid_score.calculate_fid_given_paths(
#     paths=[real_images_folder, generated_images_folder],
#     batch_size=64,
#     device=device,
#     dims=2048,           # Inception-v3 特征维度
#     num_workers=4
# )
# print(f'FID: {fid_value}')


# ========================================== #
## pip install torch-fidelity
# from torch_fidelity import calculate_metrics
#
# metrics = calculate_metrics(
#     input1="path/to/generated_images",
#     input2="path/to/real_images",
#     cuda=True,
#     fid=True,
#     is_=True,            # 同时计算 IS
#     verbose=True
# )
#
# print(f"FID: {metrics['frechet_inception_distance']}")
# print(f"IS: {metrics['inception_score_mean']}")


