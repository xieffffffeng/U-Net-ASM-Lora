import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm
from transformers import SamModel, SamProcessor
from peft import PeftModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用设备: {device}")


# ==========================================
# 1. RLE 编码函数 (Kaggle 官方标准要求)
# ==========================================
def rle_encoding(x):
    # x 应该是一个 2D 的 numpy 数组 (只包含一个细胞的 mask)
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return ' '.join([str(x) for x in run_lengths])


# ==========================================
# 2. 加载 Base 模型 + LoRA 权重
# ==========================================
base_model_path = "./sam_local/KyanChen/sam-vit-base"
lora_weights_path = "./sam_lora_weights_50"

print("正在加载基础模型与处理器...")
processor = SamProcessor.from_pretrained(base_model_path)
base_model = SamModel.from_pretrained(base_model_path)

print("正在融合 LoRA 权重...")
# 使用 PeftModel.from_pretrained 将 LoRA 外挂组合到基础模型上
model = PeftModel.from_pretrained(base_model, lora_weights_path).to(device)
model.eval()  # 开启评估模式，关闭 Dropout 等训练行为

# ==========================================
# 3. 对测试集进行推理
# ==========================================
test_dir = 'D:/0099/U-Net/stage2_test_final'
output_csv = 'sam_lora_submission.csv'

test_ids = os.listdir(test_dir)
predictions = []

print(f"\n🚀 开始对 {len(test_ids)} 张测试图片进行预测...")

for img_id in tqdm(test_ids, desc="Predicting"):
    img_folder = os.path.join(test_dir, img_id)
    img_path = glob.glob(os.path.join(img_folder, 'images', '*.png'))[0]

    # 读取图片并获取原始尺寸
    original_img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = original_img.size

    # 构造测试期的 Prompt: 一个覆盖全图的 Bounding Box
    prompt_box = [0, 0, orig_w, orig_h]

    # 预处理
    inputs = processor(
        original_img,
        input_boxes=[[[prompt_box]]],
        return_tensors="pt"
    ).to(device)

    # 模型前向传播
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    # ==========================================
    # 【核心修复区】：去除黑边与黑白取反
    # ==========================================
    # 1. 使用官方 Processor 优雅去除 Padding 并恢复原图绝对精确的比例
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )

    # 提取出的 mask 是 boolean 数组 (True/False)
    pred_mask = masks[0].squeeze().numpy()

    # 2. 将黑白颠倒取反 (背景变黑，细胞变白)
    binary_mask = (~pred_mask).astype(np.uint8)

    # ==========================================
    # 4. 后处理：连通域分析分离细胞体
    # ==========================================
    # cv2.connectedComponents 会把原本一张连成一片的 mask，拆分成一个个单独的区域
    num_labels, labels = cv2.connectedComponents(binary_mask)

    # 如果没找到细胞，给一个空的预测
    if num_labels == 1:
        predictions.append({'ImageId': img_id, 'EncodedPixels': ''})
    else:
        # label 0 是背景，所以从 1 开始遍历每一个独立的细胞
        for label_idx in range(1, num_labels):
            single_cell_mask = (labels == label_idx).astype(np.uint8)

            # 【过滤机制】：计算当前这块“细胞”的像素面积
            area = np.sum(single_cell_mask)
            # 过滤掉面积小于 10 像素的噪点，或面积大于图片总像素 50% 的巨型背景块
            if area < 10 or area > (orig_w * orig_h * 0.5):
                continue

            rle = rle_encoding(single_cell_mask)
            predictions.append({'ImageId': img_id, 'EncodedPixels': rle})

# ==========================================
# 5. 保存结果
# ==========================================
df = pd.DataFrame(predictions)
df.to_csv(output_csv, index=False)
print(f"\n🎉 大功告成！提交文件已保存至: {os.path.abspath(output_csv)}")