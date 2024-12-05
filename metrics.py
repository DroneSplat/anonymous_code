import os
import torch
import json
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import lpips
from PIL import Image
import argparse

def calculate_metrics(img1_path, img2_path, loss_fn):
    img1 = np.array(Image.open(img1_path).convert("RGB"))
    img2 = np.array(Image.open(img2_path).convert("RGB"))
    
    psnr_value = psnr(img1, img2)
    ssim_value = ssim(img1, img2, channel_axis=-1)
    
    img1_tensor = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img2_tensor = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    lpips_value = loss_fn(img1_tensor, img2_tensor).item()
    
    return psnr_value, ssim_value, lpips_value

def process_folders(rendering, gt, output):
    files1 = set(os.listdir(rendering))
    files2 = set(os.listdir(gt))
    
    common_files = files1.intersection(files2)
    
    results = []
    total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0
    count = 0
    
    loss_fn = lpips.LPIPS(net='alex')
    
    for file_name in common_files:
        img1_path = os.path.join(rendering, file_name)
        img2_path = os.path.join(gt, file_name)
        
        psnr_value, ssim_value, lpips_value = calculate_metrics(img1_path, img2_path, loss_fn)

        results.append({
            "image_name": file_name,
            "psnr": psnr_value,
            "ssim": ssim_value,
            "lpips": lpips_value
        })

        total_psnr += psnr_value
        total_ssim += ssim_value
        total_lpips += lpips_value
        count += 1
    
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

    with open(output, 'w') as f:
        json.dump(results, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Calculate PSNR, SSIM, and LPIPS between images in two folders.")
    parser.add_argument('--rendering', type=str, required=True, help='Path to the first folder containing images.')
    parser.add_argument('--gt', type=str, required=True, help='Path to the second folder containing images.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output JSON file to save results.')
    args = parser.parse_args()

    process_folders(args.rendering, args.gt, args.output)

if __name__ == "__main__":
    main()
