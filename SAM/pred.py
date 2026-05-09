import os
import glob
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from transformers import SamModel, SamProcessor
from peft import PeftModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 加载模型
print("正在加载模型...")
base_model_path = "./sam_local/KyanChen/sam-vit-base"
lora_weights_path = "./sam_lora_weights"
processor = SamProcessor.from_pretrained(base_model_path)
base_model = SamModel.from_pretrained(base_model_path)
model = PeftModel.from_pretrained(base_model, lora_weights_path).to(device)
model.eval()

# 2. 随便拿一张测试集的图片
test_dir = 'D:/0099/U-Net/stage2_test_final'
test_ids = os.listdir(test_dir)
sample_id = test_ids[0] # 取第一张图
img_path = glob.glob(os.path.join(test_dir, sample_id, 'images', '*.png'))[0]

original_img = Image.open(img_path).convert('RGB')
orig_w, orig_h = original_img.size

# 3. 进行推理
prompt_box = [0, 0, orig_w, orig_h]
inputs = processor(original_img, input_boxes=[[[prompt_box]]], return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs, multimask_output=False)

pred_masks = outputs.pred_masks.squeeze(1)
pred_masks_resized = F.interpolate(pred_masks, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
prob_mask = torch.sigmoid(pred_masks_resized).squeeze().cpu().numpy()

# 4. 画图展示
plt.figure(figsize=(12, 4))

# 子图1：原图
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(original_img)
plt.axis('off')

# 子图2：模型输出的热力图 (概率图)
plt.subplot(1, 3, 2)
plt.title("Model Prediction (Probabilities)")
plt.imshow(prob_mask, cmap='jet')
plt.colorbar(fraction=0.046, pad=0.04)
plt.axis('off')

# 子图3：最终的二值化 Mask
plt.subplot(1, 3, 3)
plt.title("Final Binary Mask (>0.5)")
plt.imshow(prob_mask > 0.5, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()