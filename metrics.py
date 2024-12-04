import os
import torch
import json
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import lpips
from PIL import Image
import argparse

def calculate_metrics(img1_path, img2_path, loss_fn):
    # 打开并转换图片为RGB
    img1 = np.array(Image.open(img1_path).convert("RGB"))
    img2 = np.array(Image.open(img2_path).convert("RGB"))
    
    # 计算 PSNR 和 SSIM
    psnr_value = psnr(img1, img2)
    ssim_value = ssim(img1, img2, channel_axis=-1)
    
    # 使用 LPIPS 模型计算感知损失
    img1_tensor = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img2_tensor = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    lpips_value = loss_fn(img1_tensor, img2_tensor).item()
    
    return psnr_value, ssim_value, lpips_value

def process_folders(folder1, folder2, output_json):
    # 获取两个文件夹中的所有图片文件名
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))
    
    # 找到文件名相同的图片
    common_files = files1.intersection(files2)
    
    results = []
    total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0
    count = 0
    
    # 初始化 LPIPS 模型
    loss_fn = lpips.LPIPS(net='alex')
    
    # 计算每对图片的指标
    for file_name in common_files:
        img1_path = os.path.join(folder1, file_name)
        img2_path = os.path.join(folder2, file_name)
        
        psnr_value, ssim_value, lpips_value = calculate_metrics(img1_path, img2_path, loss_fn)

        results.append({
            "image_name": file_name,
            "psnr": psnr_value,
            "ssim": ssim_value,
            "lpips": lpips_value
        })

        # 累计总值
        total_psnr += psnr_value
        total_ssim += ssim_value
        total_lpips += lpips_value
        count += 1
    
    # 计算平均值
    avg_psnr = total_psnr / count if count > 0 else 0
    avg_ssim = total_ssim / count if count > 0 else 0
    avg_lpips = total_lpips / count if count > 0 else 0
    
    # 将平均指标添加到结果
    results.append({
        "average_metrics": {
            "psnr": avg_psnr,
            "ssim": avg_ssim,
            "lpips": avg_lpips
        }
    })

    # 将结果写入 JSON 文件
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Calculate PSNR, SSIM, and LPIPS between images in two folders.")
    parser.add_argument('--folder1', type=str, required=True, help='Path to the first folder containing images.')
    parser.add_argument('--folder2', type=str, required=True, help='Path to the second folder containing images.')
    parser.add_argument('--output_json', type=str, required=True, help='Path to the output JSON file to save results.')
    args = parser.parse_args()

    process_folders(args.folder1, args.folder2, args.output_json)

if __name__ == "__main__":
    main()
