from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import os

# --- 配置 ---
# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前使用的设备: {device}")

# 加载模型和处理器 (只需要加载一次)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 定义存放图片的文件夹路径
# !!! 请务必将下面的路径替换成你实际存放图片的文件夹路径 !!!
image_folder_path = r"./images"

# 定义用于比较的文本列表
# 为了更好地演示效果，我添加了一个不相关的文本，这样你可以看到多个文本的比较结果
# 如果你只比较一个文本，通常直接使用点积（余弦相似度）更直观
# 如果你有多个文本需要比较，并且希望看到它们之间相对的概率，softmax 才有意义
texts_for_comparison = ["a photo of a dog","Pictures of wearing swimsuits"]

# --- 新增：温度参数 ---
# 这是一个需要根据你的数据进行实验调整的值
# 尝试 20, 50, 100 等值，看看哪个能更好地放大你想要看到的差异。
temperature_value = 50.0 # 初始值，你可以调整它来观察效果

# --- 图片处理函数 ---
def process_image_batch(image_paths, texts, model, processor, device, temperature_value):
    """
    批量处理图片，并计算它们与给定文本的相似度。

    参数:
        image_paths (list): 包含图片完整路径的列表。
        texts (list): 用于比较的文本字符串列表。
        model (CLIPModel): 已加载的 CLIP 模型。
        processor (CLIPProcessor): 已加载的 CLIP 处理器。
        device (str): 运行推理的设备 ('cuda' 或 'cpu')。
        temperature_value (float): 用于缩放相似度分数的温度参数。

    返回:
        dict: 一个字典，键是图片路径，值是另一个字典，其中包含该图片与每个文本的得分。
    """
    results = {}
    
    images = []
    valid_image_paths_in_batch = [] # 记录当前批次成功加载的图片路径
    for img_path in image_paths:
        try:
            # 打开图片并转换为 RGB 格式，确保兼容性
            img = Image.open(img_path).convert("RGB") 
            images.append(img)
            valid_image_paths_in_batch.append(img_path)
        except Exception as e:
            print(f"警告: 无法打开或处理图片 {img_path}。已跳过。错误: {e}")
            continue

    if not images:
        print("当前批次中没有找到有效的图片。跳过此批次的处理。")
        return results

    # 处理数据并迁移到设备 (CLIP 处理器会自动处理多张图片)
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)

    # 推理
    with torch.no_grad():
        outputs = model(**inputs)

    # 计算相似度
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds

    # 步骤 1: 计算原始余弦相似度 (点积)
    similarity_matrix = (image_embeds @ text_embeds.T)

    # --- 关键修改 1: 应用温度缩放 ---
    # 将原始相似度分数乘以温度值
    scaled_similarity_matrix = similarity_matrix * temperature_value

    # --- 关键修改 2: 应用 Softmax ---
    # 对缩放后的相似度应用 softmax，将其转换为概率分布
    # dim=-1 表示对每个图片的文本相似度进行归一化，使其和为1
    probability_scores = scaled_similarity_matrix.softmax(dim=-1)

    # 存储每个图片的结果
    for i, img_path in enumerate(valid_image_paths_in_batch): # 遍历成功加载的图片路径
        image_results = {}
        for j, text in enumerate(texts):
            # 现在我们存储的是经过 softmax 后的概率分数
            image_results[text] = probability_scores[i, j].item()
        results[img_path] = image_results

    return results

# --- 主批量处理逻辑 ---

# 检查文件夹是否存在
if not os.path.isdir(image_folder_path):
    print(f"错误: 指定的图片文件夹不存在: {image_folder_path}")
else:
    all_image_files = []
    # 遍历文件夹，收集所有支持的图片文件
    print(f"正在扫描文件夹: {image_folder_path}...")
    for filename in os.listdir(image_folder_path):
        # 识别常见的图片文件扩展名 (你可以根据需要添加更多)
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff",".webp")):
            all_image_files.append(os.path.join(image_folder_path, filename))

    if not all_image_files:
        print(f"在 '{image_folder_path}' 中没有找到任何图片文件。请检查文件夹和文件类型。")
    else:
        print(f"找到 {len(all_image_files)} 张图片等待处理。")

        # 一次性处理所有图片 (如果图片数量不多，这是最简单的方式)
        # 注意这里将 texts 替换为 texts_for_comparison，以便同时比较两个文本
        # --- 传递 temperature_value 参数 ---
        full_results = process_image_batch(all_image_files, texts_for_comparison, model, processor, device, temperature_value)

        print("\n--- 所有图片处理完成 ---")
        # 打印详细结果
        for img_path, scores in full_results.items():
            print(f"\n图片文件: {os.path.basename(img_path)}") # 只显示文件名，更简洁
            for text, score in scores.items():
                print(f"   '{text}': {score:.4f}")