# U-Net · SAM-LoRA — 细胞核分割

2018 Data Science Bowl 细胞核分割竞赛的双方案实现：**U-Net（从头训练）** 与 **SAM + LoRA（高效微调）**。

## 项目结构

```
├── U-Net/                          # 经典 U-Net 方案
│   ├── main.py                     # U-Net 模型定义 + 训练 + 验证
│   ├── train.py                    # DiceLoss + AttentionGate 模块
│   ├── generate_submission.py      # 生成提交文件
│   ├── make_submission.py          # 提交文件处理
│   └── submission.csv              # 预测结果
│
├── SAM/                            # SAM + LoRA 方案
│   ├── train_lora.py               # LoRA 微调训练脚本
│   ├── pred.py                     # 加载 LoRA 权重进行推理预测
│   ├── base.py                     # SAM 基础模型下载（国内镜像）
│   ├── download_model.py           # 模型下载工具
│   ├── sam_lora_weights/           # LoRA 微调权重（含 adapter_config）
│   ├── sam_lora_weights_50/        # LoRA 微调权重（另一组超参）
│   └── sam_lora_submission.csv     # SAM-LoRA 预测结果
│
├── stage1_train/                   # 竞赛 Stage 1 训练集
├── stage1_test/                    # 竞赛 Stage 1 测试集
├── stage2_test_final/              # 竞赛 Stage 2 测试集
├── stage1_sample_submission.csv    # 提交样例
├── stage1_solution.csv             # Stage 1 答案
└── stage2_sample_submission_final.csv
```

## 方案一：U-Net

经典 encoder-decoder 结构，完整从零训练。

- **网络结构**：卷积下采样 + 跳跃连接上采样
- **损失函数**：DiceLoss（配合 softmax）
- **注意力机制**：AttentionGate，抑制无关区域响应
- **训练设备**：支持 CPU / GPU

```bash
cd U-Net
python main.py                # 训练
python generate_submission.py # 生成预测
```

## 方案二：SAM + LoRA

基于 Meta 的 Segment Anything Model（SAM），使用 LoRA 高效微调，仅训练少量参数量即可适配细胞核分割任务。

- **基座模型**：`facebook/sam-vit-base`
- **微调方法**：LoRA（`r=16`, `alpha=16`，目标模块 `qkv`）
- **额外训练**：mask_decoder 全部参数参与训练（约 400 万参数）
- **冻结部分**：image_encoder、prompt_encoder 等骨干网络
- **国内镜像**：支持 `hf-mirror.com` 加速模型下载

```bash
cd SAM
python train_lora.py          # LoRA 微调训练
python pred.py                 # 加载 LoRA 权重推理
```

## 数据

竞赛数据来自 [2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018)：

- 训练集：`stage1_train/` — 多张细胞核显微镜图像 + 对应 mask
- 测试集：`stage1_test/` + `stage2_test_final/`
- 每张图像可包含多个细胞核，mask 为独立 PNG 文件

## 环境依赖

```txt
torch
torchvision
transformers
peft
Pillow
numpy
tqdm
matplotlib
```

## 注意事项

- 模型权重文件（`.safetensors`、`.pth`）超过 GitHub 100MB 限制，未包含在仓库中。运行训练脚本会自动下载或生成权重
- LoRA adapter 配置文件（`adapter_config.json`）和描述文件（`README.md`）已包含在仓库中
- SAM 基础模型需通过 `base.py` 或 `download_model.py` 下载