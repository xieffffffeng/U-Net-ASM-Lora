import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from skimage import measure
from tqdm import tqdm

# ==========================================
# 1. 定义与训练时完全一致的 U-Net 模型
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(3, 64)
        self.down2 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up1 = DoubleConv(128 + 64, 64)
        self.outc = nn.Conv2d(64, 1, 1)
        
    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x = self.up(x2)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x)
        logits = self.outc(x)
        return logits

# ==========================================
# 2. RLE 游程编码函数 (符合 Kaggle 评测标准)
# ==========================================
def rle_encoding(x):
    # 将二维矩阵转置并展平 (Kaggle要求的列优先顺序)
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return " ".join([str(x) for x in run_lengths])

# ==========================================
# 3. 主程序：加载数据、预测、生成 CSV
# ==========================================
if __name__ == '__main__':
    # ！！！请根据你的电脑实际路径修改这里 ！！！
    TEST_DIR = "D:/0099/U-Net/stage2_test_final"
    WEIGHT_PATH = "D:/0099/U-Net/U-Net/unet_best_weights.pth"
    OUTPUT_CSV = "submission4.csv"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用设备: {device}")

    # 检查权重文件是否存在
    if not os.path.exists(WEIGHT_PATH):
        raise FileNotFoundError(f"找不到权重文件 {WEIGHT_PATH}！请确保你已经运行了训练代码并保存了权重。")

    # 加载模型和权重
    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
    model.eval()  # 设置为推理模式

    test_ids = os.listdir(TEST_DIR)
    
    # 图像预处理 (和训练时保持一致的尺寸)
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor()
    ])

    submission_data = []

    print("开始预测并生成 Kaggle 提交文件...")
    for img_id in tqdm(test_ids):
        img_folder = os.path.join(TEST_DIR, img_id)
        img_path = glob.glob(os.path.join(img_folder, 'images', '*.png'))[0]
        
        # 1. 读取原图并获取原始尺寸
        original_image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = original_image.size
        
        # 2. 预处理并送入模型预测
        input_tensor = transform(original_image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)
            
        # 取出预测结果并二值化 (阈值0.5)
        pred_mask_128 = probs[0, 0].cpu().numpy()
        pred_mask_128 = (pred_mask_128 > 0.5).astype(np.uint8)

        # 3. 将 128x128 的预测图放大回原始尺寸 (重要！否则Kaggle评测报错)
        pred_mask_orig = Image.fromarray(pred_mask_128 * 255).resize((orig_w, orig_h), Image.NEAREST)
        pred_mask_orig = np.array(pred_mask_orig) / 255
        
        # 4. 实例分割：把连成一片的细胞核拆分成独立的个体
        labeled_mask = measure.label(pred_mask_orig)
        
        # 如果该图片没有预测出任何细胞核
        if labeled_mask.max() == 0:
            submission_data.append({'ImageId': img_id, 'EncodedPixels': ''})
            continue
            
        # 5. 遍历每一个独立的细胞核，分别进行 RLE 编码
        for region_id in range(1, labeled_mask.max() + 1):
            single_nucleus_mask = (labeled_mask == region_id).astype(np.uint8)
            rle_code = rle_encoding(single_nucleus_mask)
            
            # 添加到提交列表
            submission_data.append({
                'ImageId': img_id, 
                'EncodedPixels': rle_code
            })

    # 6. 保存为 CSV 文件
    df = pd.DataFrame(submission_data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n生成完毕！文件已成功保存为: {OUTPUT_CSV}")
