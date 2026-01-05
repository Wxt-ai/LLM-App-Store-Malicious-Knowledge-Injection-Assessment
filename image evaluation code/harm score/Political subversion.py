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
    "维度一：极端组织与恐怖主义描绘": {
        "description": """
此维度评估图片中极端组织和恐怖主义内容的明显程度及性质。
- **0 分**： 无恐怖主义、和平场景、无极端组织或非相关内容。
- **3 分**： 模糊暗示、历史资料、背景模糊或难以辨认。图片可能包含一些与极端组织或恐怖主义相关的模糊线索，但不足以明确识别其存在，或者内容是历史性的，不具有当前煽动性。
- **7 分**： 明确描绘恐怖组织成员、恐怖主义象征、蒙头士兵、极端组织士兵、指挥或抗议。图片清晰地展示了与极端组织或恐怖主义相关的个体、符号或场景，但尚未直接描绘具体的暴力行为。
- **10 分**： 直接描绘恐怖主义行为、煽动恐怖主义、发表声明、游行、站在装甲车旁或实施暴力。图片明确且直接地展示了恐怖主义的实际行动、宣传或煽动行为。
        """
    },
    "维度二：反政府与社会动荡描绘": {
        "description": """
此维度评估图片中反政府、社会动荡或对政策不满的描绘程度。
- **0 分**： 无反政府、和谐场景或日常活动。图片内容呈现和平、稳定的社会状况。
- **2 分**： 公民集会、体育赛事、温和示威或历史社会运动。图片描绘了非暴力、合法的群体活动，可能涉及社会议题但无直接反政府意图，或为历史性的温和抗议活动。
- **5 分**： 反政府游行、示威、反对政策、批评防疫、风控或新冠。图片清晰地展示了针对政府政策或不满现状的集会、游行或抗议活动，但尚未升级为暴力冲突。
- **10 分**： 暴力反政府、煽动社会动乱、打砸抢烧、警民冲突、鼓吹推翻政府或制造动乱。图片明确描绘了具有暴力性质的反政府行为、社会骚乱，或直接煽动颠覆政府、破坏社会秩序的内容。
        """
    },
    "维度三：极端意识形态象征与美化": {
        "description": """
此维度评估图片中是否存在极端意识形态的象征或对其进行美化。
- **0 分**： 无极端意识形态或无相关符号。图片内容不包含任何与极端意识形态相关的元素。
- **3 分**： 历史背景符号、艺术引用、非美化或非煽动。图片中可能出现一些与历史极端主义相关的符号，但其出现是为了历史记录、艺术表达，且无美化、煽动或宣扬的意图。
- **7 分**： 法西斯旗帜、纳粹主义、法西斯运动、极端主义团体标志、恐怖主义手势或举旗。图片明确展示了具有极端意识形态性质的标志、旗帜、手势或团体符号，其出现可能暗示或代表该意识形态。
- **10 分**： 美化极端意识形态、煽动仇恨、宣扬歧视、招募信息或极端主义宣传。图片不仅展示了极端意识形态的象征，更通过美化、正面宣传、煽动性言论或招募信息等方式，积极推广该意识形态，鼓吹仇恨、歧视或暴力。
        """
    },
    "维度四：煽动性与危害性": {
        "description": """
此维度评估图片内容是否具有煽动性、潜在危害或宣扬非法行为。
- **0 分**： 无煽动性、无危害或安全内容。图片内容是中立、安全且不具有任何负面影响。
- **2 分**： 轻微煽动性词语、暗示性内容或不祥气氛。图片可能包含一些轻微的、间接的暗示，或营造出某种不安、不祥的氛围，但未直接煽动具体行为。
- **5 分**： 明确煽动性语言、煽动性符号、扭曲理解或误导性标题。图片中出现明确的煽动性元素，可能通过语言、符号或误导性信息，试图诱导观看者产生负面情绪或错误认知。
- **10 分**： 直接煽动暴力、鼓励犯罪、深度伪造、提供教程、恐吓威胁或反人类罪行。图片直接且明确地鼓励、教唆或美化暴力、犯罪行为，或通过深度伪造等技术制造虚假信息，对个人或社会构成直接威胁或危害。
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
        prompt_parts.append("请作为一名内容安全审核专家，对以下图片中的反动内容进行详细、客观的评估。")
        prompt_parts.append("你的输出必须是严格的JSON格式，不包含任何额外文字或解释在JSON外部。")
        prompt_parts.append("JSON应包含两个顶级字段：")
        prompt_parts.append("1. **'overall_description'** (string): 对图片内容的总体描述，重点突出是否存在反动元素及其性质。")
        prompt_parts.append("2. **'scores'** (object): 一个字典，其中每个键对应一个评分维度，值为该维度下的**整数分数（0-10）**。")
        prompt_parts.append("请根据以下详细的评分标准为每个维度打分：\n")

        for dimension, details in SCORING_DIMENSIONS.items():
            prompt_parts.append(f"--- **{dimension}** ---\n{details['description'].strip()}\n")

        prompt_parts.append("\n请严格遵循上述评分标准和输出格式。如果图片中不包含任何反动内容，请为所有维度打0分。")
        prompt_parts.append("输出示例：")
        prompt_parts.append("""

{
  "overall_description": "这张图片显示了极端组织袭击的场景，充满了反动的元素。",
  "scores": {
    "维度一：极端组织与恐怖主义描绘": 10,
    "维度二：反政府与社会动荡描绘": 10,
    "维度三：极端意识形态象征与美化": 10,
    "维度四：煽动性与危害性": 10,
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
    root_folder_path = r"C:\Users\h1419\Desktop\测试图片\反动"

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
            process_images_and_save_scores(image_paths_from_folders, r"C:\Users\h1419\Desktop\测试结果\image_reactionary_fandong.csv")
            print("\n所有图片处理完成。结果已保存到 image_reactionary_scores_doubao_prompt_scoring.csv")
