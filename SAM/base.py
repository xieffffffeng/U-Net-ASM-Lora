from transformers import SamModel, SamProcessor
import torch
import numpy as np
import os

# 1. 设置 Hugging Face 国内镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import SamModel, SamProcessor
import torch

# 2. 定义你要下载的模型名称（你刚才可能漏掉了这一行 👇）
model_id = "facebook/sam-vit-base"

print("正在从镜像站下载/加载 SAM 模型...")

# 3. 使用定义好的 model_id 加载处理器和模型
processor = SamProcessor.from_pretrained(model_id)
model = SamModel.from_pretrained(model_id)

print("模型加载成功！")

# 接下来的 LoRA 配置代码...

from transformers import SamModel, SamProcessor
import torch

# 接下来的代码...model_id = "facebook/sam-vit-base"
processor = SamProcessor.from_pretrained(model_id)
model = SamModel.from_pretrained(model_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16, # 秩，可以理解为微调的“容量”
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"], # 针对 ViT 的注意力机制进行注入
    lora_dropout=0.1,
    bias="none"
)

lora_model = get_peft_model(model, config)
lora_model.print_trainable_parameters() # 检查可训练参数占比，通常低于 5%

# 在 Dataset 的 __getitem__ 中添加
def get_bounding_box(mask):
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    return [x_min, y_min, x_max, y_max]

# 训练时使用 processor 处理图像和框
inputs = processor(image, input_boxes=[[[bbox]]], return_tensors="pt").to(device)
outputs = lora_model(**inputs, multimask_output=False)