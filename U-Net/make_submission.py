import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T
from skimage import measure
from tqdm import tqdm

# ==========================================
# 导入你在训练代码中定义的 SimpleUNet
# 如果这段代码和训练代码在同一个文件，可以直接用
# 如果在不同文件，请 from your_train_script import SimpleUNet
# ==========================================
from main import SimpleUNet


# ==========================================
# 1. RLE 编码函数 (Kaggle 官方推荐算法)
# ==========================================
def rle_encoding(x):
    """
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as string formatted
    """
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return ' '.join([str(x) for x in run_lengths])


# ==========================================
# 2. 生成提交文件的主逻辑
# ==========================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 加载模型和权重
    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load('unet_baseline_weights.pth'))
    model.eval()  # 切换到评估模式

    # 2. 准备测试集数据路径
    test_dir = 'D:/0099/U-Net/stage1_test'  # 请修改为你的测试集路径
    test_ids = os.listdir(test_dir)

    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor()
    ])

    submission_data = []

    # 3. 开始遍历测试集
    print("开始预测并生成提交数据...")
    for img_id in tqdm(test_ids):
        img_folder = os.path.join(test_dir, img_id)
        img_path = glob.glob(os.path.join(img_folder, 'images', '*.png'))[0]

        # 读取原图，获取原始尺寸 (W, H)
        original_image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = original_image.size

        # 预处理
        input_tensor = transform(original_image).unsqueeze(0).to(device)  # 增加 batch 维度

        # 模型预测
        with torch.no_grad():
            logits = model(input_tensor)
            # 使用 Sigmoid 将输出映射到 0~1 之间
            probs = torch.sigmoid(logits)

        # 提取预测掩码 (batch=0, channel=0)，并转回 CPU numpy
        pred_mask_128 = probs[0, 0].cpu().numpy()

        # 二值化 (阈值设为 0.5)
        pred_mask_128 = (pred_mask_128 > 0.5).astype(np.uint8)

        # 把 128x128 的预测结果放大回原始图片尺寸
        pred_mask_orig = Image.fromarray(pred_mask_128 * 255).resize((orig_w, orig_h), Image.NEAREST)
        pred_mask_orig = np.array(pred_mask_orig) / 255

        # ========================================================
        # 核心逻辑：分离实例 (Instance Segmentation)
        # 将一张图上的多个细胞核区分开，变成不同的 label (1, 2, 3...)
        # ========================================================
        labeled_mask = measure.label(pred_mask_orig)

        # 如果这张图片什么都没预测出来 (没有细胞核)
        if labeled_mask.max() == 0:
            submission_data.append({'ImageId': img_id, 'EncodedPixels': ''})
            continue

        # 遍历每一个被识别出来的独立细胞核
        for region_id in range(1, labeled_mask.max() + 1):
            # 提取单个细胞核的 mask
            single_nucleus_mask = (labeled_mask == region_id).astype(np.uint8)

            # 进行 RLE 编码
            rle_code = rle_encoding(single_nucleus_mask)

            submission_data.append({
                'ImageId': img_id,
                'EncodedPixels': rle_code
            })

    # 4. 保存为 CSV
    df = pd.DataFrame(submission_data)
    df.to_csv('submission.csv', index=False)
    print("生成完毕！文件已保存为 submission.csv，可以提交到 Kaggle 了。")