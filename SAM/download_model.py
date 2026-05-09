from modelscope.hub.snapshot_download import snapshot_download

print("开始从 ModelScope 下载 SAM 基础模型到本地...")
# 【关键修复】换成了 ModelScope 上真实存在的 SAM 仓库名
model_dir = snapshot_download('KyanChen/sam-vit-base', cache_dir='./sam_local')
print(f"下载完成！模型保存在: {model_dir}")