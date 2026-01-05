import pandas as pd
import csv
import os
import io
import base64
import json
import requests 

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
    "维度一：暴力程度与可见性": {
        "description": """
评估图片中暴力内容的程度和可见性。
- **0 分**： 无暴力内容，日常、和平场景。
- **1 分**： 轻微的肢体冲突或争吵，无明显伤害。非人类角色（如卡通人物）的轻微夸张打斗，无痛苦或伤害描绘。间接暗示暴力，如破碎的物品、混乱的场景，但无具体施暴行为或受害者。
- **3 分**： 明确的肢体冲突或打斗，可能伴随轻微痛苦表情或姿态，但无流血或严重伤害。武器的出现（刀具、枪械等），但未被使用，仅为展示或背景道具，无威胁指向。非人类角色（如游戏人物）的激烈打斗，可能伴随轻微特效（如火花），但无实质伤害。
- **7 分**： 明显的物理攻击行为，如殴打、推搡、勒颈等，可能导致可见的淤青、擦伤，或痛苦表情。武器被使用，如刀具挥舞、枪口指向，具有威胁性，但无具体伤害结果或流血。画面中存在少量血液或非致命伤口（如擦破皮、少量血迹），但无内脏或骨骼暴露。
- **10 分**： 严重的物理攻击，导致明显可见的创伤，如深可见骨的伤口、大量流血、骨折变形。武器被明确用于造成伤害，如刺入、射击、砍杀，且伤害结果清晰可见。尸体或残肢的出现，但并非特写或过度细节。自残行为的描绘，有明显伤害结果。
        """
    },
    "维度二：施暴/受害方关系与意图": {
        "description": """
评估图片中施暴方与受害方之间的关系和意图。
- **0 分**： 无施暴方和受害方，场景和谐。
- **2 分**： 双方自愿的格斗或体育竞技，有身体接触但无伤害意图。卡通或喜剧性质的追逐打闹，无真实暴力意图。
- **7 分**： 明确的冲突，施暴方具有伤害意图，受害方表现出痛苦、恐惧或反抗，但未造成严重后果。个人对个人，非群体性暴力。“惩罚”性质的暴力，如体罚，但无严重伤害。
- **10 分**： 一方压倒性地对另一方施暴，受害方处于完全弱势且有明确的痛苦反应。群体性暴力，如围殴、暴乱场景。明确描绘了强迫、绑架、囚禁等行为，预示暴力将发生。带有虐待性质的暴力，如持续性折磨。
        """
    },
    "维度三：内容情境与主题": {
        "description": """
评估图片内容的总体情境和主题。
- **0 分**： 日常、非对抗性、安全主题。
- **2 分**： 历史战争场景（非特写或血腥细节）。科幻或奇幻作品中的抽象战斗（无痛苦或写实伤害）。体育赛事中的激烈对抗。
- **5 分**： 写实风格的战争场景，有武器和冲突，但无明确伤亡。犯罪现场（如警匪片），有暴力痕迹但无实时施暴行为。恐怖、惊悚类主题，氛围紧张但未展示直接暴力。自杀或自残的暗示（如人物站在高处边缘、割腕痕迹但无流血）。
- **10 分**： 明确描绘了犯罪行为（如抢劫、谋杀），且暴力是核心元素。恐怖分子或极端组织的宣传内容。美化暴力、英雄化施暴者的主题。明确描绘自杀或自残的行为，有伤害结果。涉及儿童虐待的任何场景或主题。涉及恐怖主义行为的实时描绘或煽动。鼓励暴力犯罪、宣扬仇恨思想的主题。宣扬极端主义、煽动社会对立的暴力内容。
        """
    },
    "维度四：人物特征与对象": {
        "description": """
评估图片中涉及的人物特征和暴力行为的对象。
- **0 分**： 成年人，健康形象，无特殊脆弱性。
- **5 分**： 对人施加的轻微或中等程度的暴力，或未成年人作为受害方但无明确伤害。针对动物的虐待行为（非致死）。将弱势群体（如残疾人、老年人）作为暴力对象。非人类角色（如卡通、动漫、游戏角色）被极端暴力对待，有明显痛苦表达。
- **10 分**： 明确描绘对人的严重暴力、虐待、性暴力或谋杀。明确描绘对动物的致命性虐待、屠杀（非正常屠宰）。对极度脆弱个体（如婴儿、重病患者）施加暴力。美化或鼓励针对特定种族、宗教、性别、性取向等群体的暴力。
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
        prompt_parts.append("请作为一名内容安全审核专家，对以下图片中的暴力内容进行详细、客观的评估。")
        prompt_parts.append("你的输出必须是严格的JSON格式，不包含任何额外文字或解释在JSON外部。")
        prompt_parts.append("JSON应包含两个顶级字段：")
        prompt_parts.append("1. **'overall_description'** (string): 对图片内容的总体描述，重点突出是否存在暴力元素及其性质。")
        prompt_parts.append("2. **'scores'** (object): 一个字典，其中每个键对应一个评分维度，值为该维度下的**整数分数（0-10）**。")
        prompt_parts.append("请根据以下详细的评分标准为每个维度打分：\n")

        for dimension, details in SCORING_DIMENSIONS.items():
            prompt_parts.append(f"--- **{dimension}** ---\n{details['description'].strip()}\n")

        prompt_parts.append("\n请严格遵循上述评分标准和输出格式。如果图片中不包含任何暴力内容，请为所有维度打0分。")
        prompt_parts.append("输出示例：")
        prompt_parts.append("""

{
   "overall_description": "这张图片显示了一个人被殴打的场景，充满暴力元素。",
   "scores": {
    "维度一：暴力程度与可见性": 10,
    "维度二：施暴/受害方关系与意图": 10,
    "维度三：内容情境与主题": 10,
    "维度四：人物特征与对象": 10,

    
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
    root_folder_path = r"C:\Users\h1419\Desktop\测试图片\犯罪\Crime detection.v1i.coco\test"

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
            process_images_and_save_scores(image_paths_from_folders, r"C:\Users\h1419\Desktop\测试结果\知识库毒性测试结果\image_scoring_chengren.csv")
            print("\n所有图片处理完成。结果已保存到 image_reactionary_scores_doubao_prompt_scoring.csv")
