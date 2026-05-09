import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm import tqdm


# ==========================================
# 1. 定义数据集 (合并多个 Mask)
# ==========================================
class DSBowlDataset(Dataset):
    def __init__(self, root_dir, img_size=256):
        self.root_dir = root_dir
        # 获取所有样本的文件夹 ID
        self.image_ids = os.listdir(root_dir)
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_folder = os.path.join(self.root_dir, img_id)

        # 1. 读取原图 (取 RGB)
        img_path = glob.glob(os.path.join(img_folder, 'images', '*.png'))[0]
        image = Image.open(img_path).convert('RGB')

        # 2. 读取并合并所有的 Masks
        mask_paths = glob.glob(os.path.join(img_folder, 'masks', '*.png'))
        # 初始化一个全黑的 numpy 数组，尺寸和原图一致
        combined_mask = np.zeros((image.size[1], image.size[0]), dtype=np.float32)

        for mp in mask_paths:
            mask = np.array(Image.open(mp).convert('L'))
            combined_mask = np.maximum(combined_mask, mask)  # 将所有细胞核叠加到一张图上

        combined_mask = Image.fromarray(combined_mask).convert('L')

        # 3. 转换为 Tensor
        image = self.transform(image)
        mask = self.transform(combined_mask)
        # 将 mask 二值化 (0 和 1)
        mask = (mask > 0).float()

        return image, mask


# ==========================================
# 2. 极简版 U-Net 模型
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
        # 编码器 (下采样)
        self.down1 = DoubleConv(3, 64)
        self.down2 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)

        # 解码器 (上采样)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up1 = DoubleConv(128 + 64, 64)

        # 输出层 (1个通道，用于二分类预测细胞核)
        self.outc = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))

        x = self.up(x2)
        x = torch.cat([x, x1], dim=1)  # 跳跃连接 (Skip connection)
        x = self.up1(x)
        logits = self.outc(x)
        return logits


# ==========================================
# 3. 训练主循环
# ==========================================
# ==========================================
# 3. 训练主循环
# ==========================================
if __name__ == '__main__':
    # 配置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用设备: {device}")

    # 数据准备
    train_dir = 'D:/0099/U-Net/stage1_train'
    if not os.path.exists(train_dir):
        raise FileNotFoundError("找不到数据！请先运行 prepare_data.py 解压数据。")

    dataset = DSBowlDataset(root_dir=train_dir, img_size=128)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    # 初始化模型、损失函数和优化器
    model = SimpleUNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 开始训练
    epochs = 100

    # ！！！新增：定义一个变量来记录历史最低的 Loss ！！！
    best_loss = float('inf')  # 初始值设为无穷大

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # 使用 tqdm 显示进度条
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}')
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # 计算这个 Epoch 的平均 Loss
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} 平均 Loss: {avg_epoch_loss:.4f}")

        # ！！！新增：判断并保存最佳权重的逻辑 ！！！
        if avg_epoch_loss < best_loss:
            print(f"--> 模型改善啦！平均 Loss 从 {best_loss:.4f} 降到了 {avg_epoch_loss:.4f}。保存当前权重！")
            best_loss = avg_epoch_loss
            # 保存最好的模型权重
            torch.save(model.state_dict(), 'unet_best_weights.pth')

    print("Baseline 训练完成！最好的模型权重已保存为 unet_best_weights.pth。")

