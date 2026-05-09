import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import SamModel, SamProcessor
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import torch.nn.functional as F

# 确保使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用设备: {device}")

# ==========================================
# 1. 初始化本地模型与 PEFT (LoRA)
# ==========================================
local_model_path = "./sam_local/KyanChen/sam-vit-base"
processor = SamProcessor.from_pretrained(local_model_path)
model = SamModel.from_pretrained(local_model_path)

# 冻结模型主干参数
for param in model.parameters():
    param.requires_grad = False

# 配置 LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["qkv"], # <--- SAM 的注意力层名称是 qkv
    lora_dropout=0.1,
    bias="none"
)

# 获取封装了 LoRA 的模型
peft_model = get_peft_model(model, lora_config).to(device)

# ==========================================
# 【关键修复2】：手动解冻 mask_decoder 的参数
# 因为我们没让 peft 去管 decoder，所以我们要手动把它的梯度打开
# ==========================================
for name, param in peft_model.named_parameters():
    if "mask_decoder" in name:
        param.requires_grad = True

print("\n模型参数状态：")
# 你会发现这里的 trainable params 比之前多了约 400万，因为 decoder 完整参与训练了
peft_model.print_trainable_parameters()


# ==========================================
# 2. 构建专用于 SAM 的 Dataset
# ==========================================
class SAMDataset(Dataset):
    def __init__(self, root_dir, processor):
        self.root_dir = root_dir
        self.processor = processor
        self.image_ids = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_folder = os.path.join(self.root_dir, img_id)

        # 1. 读取原图
        img_path = glob.glob(os.path.join(img_folder, 'images', '*.png'))[0]
        image = Image.open(img_path).convert('RGB')

        # 2. 读取所有的 Masks 并寻找 Bounding Box
        mask_paths = glob.glob(os.path.join(img_folder, 'masks', '*.png'))

        combined_mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
        for mp in mask_paths:
            mask = np.array(Image.open(mp).convert('L'))
            combined_mask = np.maximum(combined_mask, mask)

        ground_truth_mask = combined_mask > 0

        y_indices, x_indices = np.where(ground_truth_mask)
        if len(x_indices) == 0:  # 处理全黑图片
            prompt_box = [0, 0, image.size[0], image.size[1]]
        else:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            prompt_box = [x_min, y_min, x_max, y_max]

        # 3. 使用 Processor 进行格式转换
        inputs = self.processor(
            image,
            input_boxes=[[[prompt_box]]],
            return_tensors="pt"
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = torch.tensor(ground_truth_mask, dtype=torch.float32).unsqueeze(0)
        return inputs


# ==========================================
# 3. 训练循环 (Training Loop)
# ==========================================

# 1. 准备数据加载器
train_dir = 'D:/0099/U-Net/stage1_train'
train_dataset = SAMDataset(root_dir=train_dir, processor=processor)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# 2. 设置优化器和损失函数
optimizer = optim.AdamW(peft_model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

# 3. 开始训练
epochs = 50
print("\n🚀 开始大模型 LoRA 微调训练...")

for epoch in range(epochs):
    peft_model.train()
    epoch_loss = 0

    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}')
    for batch in progress_bar:
        # 将数据移至显卡
        pixel_values = batch["pixel_values"].to(device)
        input_boxes = batch["input_boxes"].to(device)
        gt_masks = batch["ground_truth_mask"].to(device)

        # 前向传播
        outputs = peft_model(
            pixel_values=pixel_values,
            input_boxes=input_boxes,
            multimask_output=False
        )

        pred_masks = outputs.pred_masks.squeeze(1)
        original_size = gt_masks.shape[-2:]

        pred_masks_resized = F.interpolate(
            pred_masks,
            size=original_size,
            mode="bilinear",
            align_corners=False
        )

        # 计算损失
        loss = criterion(pred_masks_resized, gt_masks)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    print(f"Epoch {epoch + 1} 平均 Loss: {epoch_loss / len(train_dataloader):.4f}")

# 4. 保存模型
save_path = "./sam_lora_weights_50"
peft_model.save_pretrained(save_path)
print(f"\n🏆 训练完成！LoRA 权重已成功保存至: {save_path}")