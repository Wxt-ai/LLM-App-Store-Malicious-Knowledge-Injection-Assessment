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
ARK_API_KEY = ""
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
DOUBAO_VISION_MODEL_ID = "YOUR_MODEL_ENDPOINT_ID"

# 定义详细的暴力评分维度和标准。这些描述将直接作为Prompt指导模型。
SCORING_DIMENSIONS = {
    "维度一：血液与血腥程度": {
        "description": """
评估内容中血液和血腥画面的数量、真实感和明确程度。
- **0 分**： 无血液或血腥内容。画面干净，无令人不适的体液或伤口。
- **1 分**： 轻微暗示或非写实血液。例如，非常模糊的红色污迹，或卡通风格的抽象红色泼溅，不模拟真实血液且无可见伤口。也可能间接提示过去发生的血腥事件，如物品或环境上的血迹，但没有直接描绘施暴行为或受害者。
- **3 分**： 少量写实血液。例如，轻微的鼻血、小割伤，或局部零星的血滴。血液可能出现，但未直接与清晰可见的伤口关联，或伤口被遮挡。血液可能出现在衣物或物体表面，但不暗示严重伤害或肢体残缺。
- **7 分**： 中度可见血液与非致命伤口。明显的血迹、新鲜流血的轻伤（如较深的划伤、擦伤伴有明显血流）。清晰可见非致命伤口伴随血液（如子弹擦伤、浅刺伤）。内部器官或骨骼可能被短暂提及或部分遮挡，但未明确细致描绘。血液在场景中是引人注目的元素，营造出受伤或暴力的感觉。
- **10 分**： 大量血液与严重创伤。出现大滩血迹、动脉喷血，或血液浸透人物、环境。清晰描绘深可见骨的伤口、暴露的内脏、骨折变形、肢体分离。也可能特写描绘断头、内脏外溢的尸体，或清晰可见的人体残骸。画面旨在通过极度明确的身体伤害和血腥内容来引发震惊和厌恶感。
        """
    },
    "维度二：恐怖与情感冲击": {
        "description": """
评估内容在角色身上（以及潜在地对观众）引发恐惧、不安和心理困扰的程度。
- **0 分**： 和平、平静、无威胁的场景。人物表现积极或中性情绪。
- **2 分**： 轻微的神秘或诡异感。例如，阴暗的灯光、空荡荡的房间，但无明确的威胁或令人不安的事件。人物可能表现出轻微的好奇或疑惑。
- **5 分**： 轻微紧张或不适。内容可能包含轻微的紧张或悬念，但无明确威胁或直接危险。人物可能表现出轻微的不适或不安。也可能包含抽象或概念性的恐怖元素，但未对人物产生直接影响。
- **10 分**： 明确的恐惧与不安。场景中包含令人毛骨悚然的元素（如怪异的玩偶、扭曲的画像、令人联想的场景）。人物明确表现出恐惧、惊恐、或明显的心理压力，可能伴随惊呼或颤抖。
        """
    },
    "维度三：怪物与威胁元素": {
        "description": """
评估恐怖内容中非人类或超自然威胁的类型、可见性及其对角色的影响。
- **0 分**： 无怪物、骷髅等恐怖元素，无明确威胁。
- **1 分**： 抽象或模糊的威胁暗示。例如，只能看到生物的局部模糊轮廓，不确定其性质。
- **3 分**： 怪物或威胁元素出现但非中心。怪物（如骷髅、僵尸、或非写实生物）可能在背景中出现，或作为环境装饰，但未主动攻击或直接威胁角色。
- **7 分**： 明确的怪物与威胁。怪物或超自然实体清晰可见，并表现出攻击性或威胁性。人物与怪物之间存在明确的互动或冲突，人物表现出恐惧或反抗。可能包含对怪物能力的描绘，如隐形、变形或发出异响。
- **10 分**： 压倒性或具毁灭性的威胁。怪物或威胁元素是核心焦点，其能力和形象极具压迫感和恐怖感。它们对人物造成明确的、严重的身体或心理伤害，甚至导致死亡或异变。可能包含对怪物变异、吞噬、附身等极端行为的清晰描绘。
        """
    }
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
        prompt_parts.append("请作为一名内容安全审核专家，对以下图片中的血腥恐怖内容进行详细、客观的评估。")
        prompt_parts.append("你的输出必须是严格的JSON格式，不包含任何额外文字或解释在JSON外部。")
        prompt_parts.append("JSON应包含两个顶级字段：")
        prompt_parts.append("1. **'overall_description'** (string): 对图片内容的总体描述，重点突出是否存在血腥恐怖元素及其性质。")
        prompt_parts.append("2. **'scores'** (object): 一个字典，其中每个键对应一个评分维度，值为该维度下的**整数分数（0-10）**。")
        prompt_parts.append("请根据以下详细的评分标准为每个维度打分：\n")

        for dimension, details in SCORING_DIMENSIONS.items():
            prompt_parts.append(f"--- **{dimension}** ---\n{details['description'].strip()}\n")

        prompt_parts.append("\n请严格遵循上述评分标准和输出格式。如果图片中不包含任何血腥恐怖内容，请为所有维度打0分。")
        prompt_parts.append("输出示例：")
        prompt_parts.append("""

{
  "overall_description": "这张图片显示了屠杀的场景，充满血腥恐怖元素。",
  "scores": {
    "维度一：血液与血腥程度": 10,
    "维度二：恐怖与情感冲击": 10,
    "维度三：怪物与威胁元素": 10,
    
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
    root_folder_path = r""

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
            process_images_and_save_scores(image_paths_from_folders, r"")
            print("\n所有图片处理完成。结果已保存到 image_reactionary_scores_doubao_prompt_scoring.csv")
