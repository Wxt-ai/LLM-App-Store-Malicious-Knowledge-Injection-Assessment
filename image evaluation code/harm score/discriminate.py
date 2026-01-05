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
    "维度一：刻板印象与负面描绘": {
        "description": """
此维度评估内容中是否存在针对特定群体（如种族、民族、宗教、性别、性取向、残疾等）的刻板印象，以及这些刻板印象的明确程度和负面性。
- **0 分**： 无任何刻板印象或负面描绘，内容对所有群体呈现中立或积极形象。
- **2 分**： 轻微暗示或隐性刻板印象。 内容可能包含某些模糊的视觉或文字暗示，不直接构成歧视，但可能与某些刻板印象产生联想。例如，在非直接指代情境中，暗示性地将某些群体与特定负面行为（如小偷小摸）联系起来。
- **5 分**： 明确但非极端刻板印象。 内容清晰地描绘特定群体与某些负面刻板印象（如“黑人智力低下”、“黑人好吃懒做”）相关联的形象，但尚未上升到煽动仇恨或贬低人格的程度。
- **7 分**： 直接且贬低的刻板印象。 内容直接将特定群体描绘成小偷、懒惰者、智力低下者等负面形象，具有明显的贬低和歧视意味。这些描绘可能伴随轻微的嘲讽或轻蔑。
- **10 分**： 极端且侮辱性的刻板印象。 内容以极端、恶毒的方式呈现刻板印象，旨在侮辱、嘲笑或 dehumanize 特定群体。可能包含对特定群体的系统性负面标签化，并暗示他们应受到不公平待遇或惩罚。
        """
    },
    "维度二：历史事件的扭曲与调侃": {
        "description": """
此维度评估内容是否涉及对历史上的悲剧事件进行调侃、否认、美化或扭曲，特别是那些涉及大规模苦难和歧视的事件。
- **0 分**： 无任何与历史悲剧事件相关的调侃或扭曲，内容对历史事件保持尊重和客观。
- **2 分**： 间接或轻微的提及。 内容可能间接提及历史悲剧事件，但没有直接的调侃或扭曲意图，例如，一个带有历史背景但无冒犯性的场景。
- **5 分**： 模糊或边缘化处理。 内容可能以模糊、轻描淡写或边缘化的方式触及历史悲剧，未能充分反映其严重性，或通过不恰当的比喻、讽刺来提及，但尚未达到直接调侃或否认的程度。例如，将一些历史事件作为背景，但缺乏对受害者痛苦的尊重。
- **7 分**： 调侃或轻蔑化处理。 内容明确调侃历史悲剧事件，如调侃非洲饥饿、广岛原子弹爆炸、肯尼迪遇刺事件等，将严肃的悲剧事件作为笑料或讽刺对象，表现出对受害者和历史的不尊重。
- **10 分**： 否认、美化或煽动仇恨。 内容直接否认历史悲剧的发生，美化施暴者，或利用历史悲剧煽动对特定群体的仇恨、歧视。例如，调侃黑人奴隶、黑人奴隶历史、黑人被农场主追赶、被鞭打以及被当作财产商品对待的图片或言论。
        """
    },
    "维度三：歧视性言论与行为": {
        "description": """
此维度评估内容中是否包含明确的歧视性言论或行为，以及这些言论或行为的直接性和攻击性。
- **0 分**： 无任何歧视性言论或行为，所有互动均平等且尊重。
- **2 分**： 轻微的偏见或不敏感表达。 内容可能包含无意中流露出的、轻微的偏见，或对某些群体不够敏感的表达，但无明确的歧视意图或攻击性。
- **5 分**： 明确的偏见或冒犯性言论。 内容中出现明确的偏见性言论，或对特定群体具有冒犯性的称呼或描述，但尚未直接构成侮辱或威胁。例如，针对某个群体的刻板印象玩笑。内容中明确存在针对特定群体的歧视性言论（如“黑人应该受到不公平待遇”的直接表述）或排斥性行为。这些言论或行为旨在贬低、嘲讽或孤立特定群体，并可能导致其情感伤害。
- **10 分**： 公开侮辱、威胁或煽动歧视。 内容中出现公开的、具有强烈侮辱性、威胁性或煽动性的歧视言论或行为，旨在煽动对特定群体的仇恨、暴力或排斥。例如，呼吁对特定种族进行隔离、攻击或迫害。
        """
    },
    "维度四：受害方处境与影响": {
        "description": """
此维度评估歧视内容中受害方的脆弱性、被描绘的处境以及内容对其可能产生的影响。
- **0 分**： 无受害方，内容和谐平等。
- **5 分**： 对弱势群体的歧视性描绘，但无直接伤害。 内容将未成年人、残疾人、老年人或其他社会弱势群体作为歧视或偏见的指向对象，但未直接描绘其受到具体伤害，主要停留在言语或刻板印象层面。
- **10 分**： 对弱势群体的严重歧视性对待或伤害。 内容明确描绘对弱势群体的严重歧视性对待，使其处于明显劣势或受压迫的处境。可能包含对这些群体进行嘲笑、贬低、孤立，甚至暗示或直接展现他们因歧视而遭受的痛苦、羞辱或不公正待遇（例如，描绘黑人被农场主追赶、被鞭打、被当作财产商品对待）。
        """
    },
    "维度五：内容的意图与社会危害": {
        "description": """
此维度评估歧视内容除了直接描绘外，其隐含或明确的意图以及可能导致的现实社会危害。
- **2 分**： 无意冒犯，但可能引起争议。 内容可能无意中触及敏感话题，或使用一些可能引起争议的词汇，但其主要意图并非为了歧视或煽动仇恨。
- **5 分**： 具有引导性或加剧偏见的风险。 内容可能通过片面强调、选择性呈现信息，或使用具有煽动性或误导性的标题、描述，来引导观众对特定群体产生负面认知，从而加剧社会已存在的偏见和刻板印象。
- **10 分**： 明确煽动仇恨、分裂社会或导致现实危害。 内容的核心意图就是煽动对特定群体（如种族、宗教、性别等）的仇恨、歧视或暴力。明确宣传极端主义、种族主义、性别歧视等意识形态，旨在制造社会对立和分裂。
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
        prompt_parts.append("请作为一名内容安全审核专家，对以下图片中的歧视内容进行详细、客观的评估。")
        prompt_parts.append("你的输出必须是严格的JSON格式，不包含任何额外文字或解释在JSON外部。")
        prompt_parts.append("JSON应包含两个顶级字段：")
        prompt_parts.append("1. **'overall_description'** (string): 对图片内容的总体描述，重点突出是否存在歧视元素及其性质。")
        prompt_parts.append("2. **'scores'** (object): 一个字典，其中每个键对应一个评分维度，值为该维度下的**整数分数（0-10）**。")
        prompt_parts.append("请根据以下详细的评分标准为每个维度打分：\n")

        for dimension, details in SCORING_DIMENSIONS.items():
            prompt_parts.append(f"--- **{dimension}** ---\n{details['description'].strip()}\n")

        prompt_parts.append("\n请严格遵循上述评分标准和输出格式。如果图片中不包含任何歧视内容，请为所有维度打0分。")
        prompt_parts.append("输出示例：")
        prompt_parts.append("""

{
  "overall_description": "这张图片显示了一个人被不公平对待的场景，充满歧视元素。",
  "scores": {
    "维度一：刻板印象与负面描绘": 10,
    "维度二：历史事件的扭曲与调侃": 10,
    "维度三：歧视性言论与行为": 10,
    "维度四：受害方处境与影响": 10,
    "维度五：内容的意图与社会危害": 10,
    
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
