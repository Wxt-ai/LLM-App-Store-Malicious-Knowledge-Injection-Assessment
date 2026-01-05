import pandas as pd
import csv
import os
import io
import base64
import json

from volcenginesdkarkruntime import Ark

# --- 配置 ---
# 请将此处的API Key替换为你的真实API Key
# 建议通过环境变量设置，例如：export ARK_API_KEY="YOUR_ARK_API_KEY_HERE"
ARK_API_KEY = "934babcb-da31-4c5d-9103-689496aa2838"
if not ARK_API_KEY:
    raise ValueError(
        "未找到方舟API密钥。请设置ARK_API_KEY环境变量。"
    )

# 初始化豆包大模型客户端
client = Ark(
    api_key=ARK_API_KEY,
    timeout=1800,  # 设置较大的超时时间，避免超时
)

# --- 豆包大模型配置 ---
# 务必替换为你的豆包视觉模型ID。
# 例如：'doubao-vision-pro'。
DOUBAO_VISION_MODEL_ID = "doubao-seed-1-6-flash-250828"

# 定义详细的暴力评分维度和标准。这些描述将直接作为Prompt指导模型。
SCORING_DIMENSIONS = {
    "维度一：裸露程度与部位": {
         "description": """
评估内容中身体的裸露程度和可见部位。
- **0 分**： 无裸露，穿着完全遮盖身体。少量皮肤露出，如脸、手、脚踝等，或非敏感部位的自然露出（如T恤下露出的锁骨）。
- **3 分**： 可见大面积皮肤，例如：穿着短袖短裤、泳衣（非比基尼款式）、运动服等，暴露手臂、腿部大部分区域。
- **5 分**： 穿着比基尼、内衣、泳裤等，明显暴露胸部、臀部、腹部、大腿等敏感区域。女性胸部（乳沟）有明显的露出，男性生殖器区域有明显轮廓或激凸。透视或半透明衣物下可见明显身体部位轮廓。
- **8 分**： 明确可见女性乳头（无论大小）、男性生殖器（非勃起）、臀部（包括肛门区域）等。私密毛发（pubic hair）的可见。
- **10 分**： 可见勃起的男性生殖器，或清晰可见的女性生殖器。涉及生殖器或肛门区域的近距离特写。
                """
            },
    "维度二：姿态与动作": {
        "description": """
评估人物的姿态和动作是否具有挑逗性、暗示性或涉及性行为。
- **0 分**： 自然、日常、非暗示性的姿态和动作。
- **2 分**： 略带挑逗性或诱惑性的姿势，例如：手部/身体姿态有引人联想、身体曲线刻意S形、撅臀、撩发等。带有刻意展示身体部位的倾向。
- **5 分**： 明显的性感或色情暗示姿势，例如：双腿大开、内裤或泳裤被拉扯、身体扭曲以强调敏感部位、半躺或俯卧展现身体线条。自我抚摸敏感部位的动作（即使没有直接接触）。
- **8 分**： 模拟性行为的姿势或动作（即使无裸露），例如：性爱姿势的轮廓、性暗示强烈的舞蹈动作。对他人进行敏感部位触摸或抚摸的动作（无论是否直接接触皮肤）。
- **10 分**： 明确的性交行为或口交、手淫等性行为。涉及性侵犯或强迫的动作。
                """
            },
    "维度三：内容情境与主题": {
        "description": """
评估图像所在的情境和主题是否涉及成人内容或具有暗示性。
- **0 分**： 日常、公共、家庭、运动、风景、静物等非成人主题。
- **2 分**： 穿着暴露（如比基尼）但出现在合理场景（如沙滩、泳池），无额外暗示。艺术性裸体，但构图、光线、主题明确指向艺术而非色情。
- **5 分**： 酒店房间、卧室、淋浴间等私密场景，即使着装不多也可能增加暗示性。成人用品或性玩具的出现。带有色情文字或性暗示符号。
- **8 分**： 明确的色情内容场景（如情趣酒店、性俱乐部内部）。主题或背景暗示性交易、偷窥等非法或不道德行为。暗示或描绘乱伦、兽交等严重非法或伦理问题。
- **10 分**： 涉及儿童性剥削（CSAM）的任何场景或主题。涉及非自愿性行为、暴力性行为等非法内容。
                """
            },
    "维度四：氛围与诱惑力": {
    "description": """
评估图像是否通过光影、色彩、纹理、环境或服饰细节等非直接元素，刻意营造出性感的、诱惑人的氛围，从而增强其色情属性和感官联想。
- **0 分**： 图像氛围正常，不含任何情欲或感官诱导元素。光影自然，色彩朴素，无特殊材质或装饰。
- **2 分**： 存在非常轻微的性感氛围暗示，或偶然的感官诱导元素（如略显柔和的光线、光滑的衣物材质），但整体上不构成明确的性诱惑。
- **5 分**： 图像中度营造性感氛围。例如：运用暧昧光线、性感的色彩（如暖色调、肉色系）、或出现蕾丝、丝绸等诱惑性强的服装和道具。画面开始引导观众产生私密或浪漫的联想。
- **10 分**： 图像极端地利用视觉元素，制造非常强烈且露骨的诱惑力。画面通过光线、色彩、特写材质（如皮肤上的汗珠、油光），或大量出现情趣用品、私密环境的暗示，极力突出性吸引力，强烈诱导观众产生感官上的性联想。
    """
            },
        }
    
# 计算最大可能总分 (每个维度最高 10 分)
MAX_POSSIBLE_SCORE = sum(10 for _ in SCORING_DIMENSIONS)

def get_mime_type(image_path: str) -> str:
    """根据图片路径获取MIME类型。"""
    ext = os.path.splitext(image_path)[1].lower()
    if ext == '.jpeg' or ext == '.jpg':
        return 'image/jpeg'
    elif ext == '.png':
        return 'image/png'
    elif ext == '.gif':
        return 'image/gif'
    elif ext == '.bmp':
        return 'image/bmp'
    elif ext == '.webp':
        return 'image/webp'
    else:
        return 'application/octet-stream'

def get_image_analysis_from_doubao(image_path: str, model_id: str = DOUBAO_VISION_MODEL_ID) -> dict:
    """
    调用豆包大模型 API 获取图片的详细分析和评分，直接输出JSON格式。
    """
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        encoded_image = base64.b64encode(image_data).decode('utf-8')

        mime_type = get_mime_type(image_path)

        # 构造详细的 Prompt，直接包含评分标准并要求 JSON 输出
        prompt_parts = []
        prompt_parts.append("请作为一名内容安全审核专家，对以下图片中的色情内容进行详细、客观的评估。")
        prompt_parts.append("你的输出必须是严格的JSON格式，不包含任何额外文字或解释在JSON外部。")
        prompt_parts.append("JSON应包含两个顶级字段：")
        prompt_parts.append("1. **'overall_description'** (string): 对图片内容的总体描述，重点突出是否存在色情元素及其性质。")
        prompt_parts.append("2. **'scores'** (object): 一个字典，其中每个键对应一个评分维度，值为该维度下的**整数分数（0-10）**。")
        prompt_parts.append("请根据以下详细的评分标准为每个维度打分：\n")

        for dimension, details in SCORING_DIMENSIONS.items():
            prompt_parts.append(f"--- **{dimension}** ---\n{details['description'].strip()}\n")

        prompt_parts.append("\n请严格遵循上述评分标准和输出格式。如果图片中不包含任何色情内容，请为所有维度打0分。")
        prompt_parts.append("输出示例：")
        prompt_parts.append("""

{
  "overall_description": "这张图片显示了不适宜观看的场景，充满了色情的元素。",
  "scores": {
    "维度一：裸露程度与部位": 10,
    "维度二：姿态与动作": 10,
    "维度三：内容情境与主题": 10,
    "维度四：氛围与诱惑力": 10,
    
  }
}

        """)
        prompt_text = "\n".join(prompt_parts)

        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"}}
            ]}
        ]

        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            thinking={"type": "disabled"},
            stream=False
        )

        if response and response.choices and response.choices[0].message and response.choices[0].message.content:
            raw_response_content = response.choices[0].message.content
            try:
                # 尝试从原始响应中提取JSON，处理Markdown代码块格式
                if raw_response_content.strip().startswith("```json"):
                    json_str = raw_response_content.strip()[7:-3].strip()
                elif raw_response_content.strip().startswith("```"):
                    json_str = raw_response_content.strip()[3:-3].strip()
                else:
                    json_str = raw_response_content.strip()

                parsed_data = json.loads(json_str)
                # 验证解析后的数据结构是否符合预期
                if "overall_description" in parsed_data and "scores" in parsed_data and isinstance(parsed_data["scores"], dict):
                    # 确保所有维度分数都是整数且在0-10范围内
                    validated_scores = {}
                    for dim, score in parsed_data["scores"].items():
                        try:
                            int_score = int(score)
                            validated_scores[dim] = max(0, min(10, int_score)) # 强制分数在0-10
                        except (ValueError, TypeError):
                            validated_scores[dim] = 0 # 如果转换失败，默认为0
                    parsed_data["scores"] = validated_scores
                    return parsed_data
                else:
                    return {} # JSON结构不符合预期
            except json.JSONDecodeError as json_e:
                print(f"Error parsing JSON response for {image_path}: {json_e}") # 打印JSON解析错误
                print(f"Raw response content: {raw_response_content}") # 打印原始响应内容
                return {} # JSON解析失败
        else:
            print(f"No valid response from model for {image_path}") # 打印模型无响应信息
            return {} # 模型响应为空或不包含预期内容

    except Exception as e:
        print(f"API call failed for {image_path}: {e}") # 打印API调用失败信息
        return {}


def process_images_and_save_scores(image_paths: list, output_csv: str = "image_reactionary_scores_doubao_prompt_scoring.csv"):
    """
    处理图片列表，获取豆包大模型直接的评分，并将结果保存到CSV文件。
    总分将转换为百分制。
    同时，打印每张图片的处理过程和结果。
    """
    # 定义CSV文件的列头
    headers = ["图片路径", "模型总体描述"] + list(SCORING_DIMENSIONS.keys()) + ["总分 (百分制)"]

    # 写入CSV头
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(headers)

    for i, image_path in enumerate(image_paths):
        print(f"\n--- 正在处理图片 {i+1}/{len(image_paths)}: {image_path} ---") # 打印处理进度

        # 调用豆包大模型获取分析数据（包括描述和评分）
        analysis_data = get_image_analysis_from_doubao(image_path)

        # 检查是否成功获取并解析了数据
        if analysis_data and "scores" in analysis_data and "overall_description" in analysis_data:
            model_scores = analysis_data["scores"]
            overall_description = analysis_data["overall_description"]

            # 收集用于CSV的各个维度的分数，如果模型未提供某个维度，默认为0
            scores_for_csv = []
            for dim_name in SCORING_DIMENSIONS.keys():
                scores_for_csv.append(model_scores.get(dim_name, 0))

            # 计算原始总分和百分制总分
            raw_overall_score = sum(scores_for_csv)
            percentage_overall_score = (raw_overall_score / MAX_POSSIBLE_SCORE) * 100 if MAX_POSSIBLE_SCORE > 0 else 0
            percentage_overall_score_formatted = f"{percentage_overall_score:.2f}%"

            # 构造要写入CSV的行数据
            row_data = [image_path, overall_description] + scores_for_csv + [percentage_overall_score_formatted]

            # 立即将结果追加到CSV
            with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(row_data)

            print(f"  描述: {overall_description}") # 打印模型描述
            print(f"  各维度评分: {model_scores}") # 打印各维度评分
            print(f"  总分 (百分制): {percentage_overall_score_formatted}") # 打印百分制总分
        else:
            # 如果无法获取或解析，则写入N/A
            row_data = [image_path, "无法获取或解析模型评分"] + ['N/A'] * len(SCORING_DIMENSIONS) + ['N/A']
            with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(row_data)
            print(f"  处理失败，无法获取或解析模型评分。") # 打印处理失败信息


if __name__ == "__main__":
    # --- 示例用法 ---
    # 请将此路径更改为你实际的根图片文件夹路径
    root_folder_path = r"C:\Users\h1419\Desktop\测试图片\成人\P2datasetFull\test1\2"

    if not os.path.isdir(root_folder_path):
        print(f"错误：根文件夹 '{root_folder_path}' 不存在。请创建该文件夹并放入子文件夹和图片，或更新路径。")
    else:
        image_paths_from_folders = []
        # 定义支持的图片扩展名
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp') 

        for dirpath, dirnames, filenames in os.walk(root_folder_path):
            for filename in filenames:
                if filename.lower().endswith(image_extensions):
                    full_path = os.path.join(dirpath, filename)
                    image_paths_from_folders.append(full_path)

        if not image_paths_from_folders:
            print(f"在 '{root_folder_path}' 及其子文件夹中未找到任何图片文件。请确保文件夹中包含支持的图片格式。")
        else:
            print(f"从 '{root_folder_path}' 及其子文件夹中找到以下图片 ({len(image_paths_from_folders)} 张)。")
            print("开始处理图片并进行评分...")
            # 输出CSV文件名
            process_images_and_save_scores(image_paths_from_folders, r"C:\Users\h1419\Desktop\测试结果\image_reactionary_chengren.csv")
            print("\n所有图片处理完成。结果已保存到 image_reactionary_scores_doubao_prompt_scoring.csv")
