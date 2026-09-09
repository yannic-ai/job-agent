from langchain_core.prompts import ChatPromptTemplate

SECTION_KEYS = (
    "title",
    "personal_info",
    "skills",
    "work_experience",
    "projects",
    "education",
    "other",
)

SYSTEM_PROMPT = """你是中文简历解析器。只根据用户提供的已切分区块填写 Resume schema，用 JSON 输出，不编造。
规则：
- 原文没有的信息用 null 或空数组。
- 明显占位（XXX、保密）视为缺失。
- personal_info.name 必须取标题区 `#` 后的姓名，不能因为基本信息里没单独写「姓名」就填 null。
- location 取期望城市或现居地原文，不要补全省市区。
- phone 取基本信息里的电话，只保留数字含义，不要把「XXX」当电话。
- 工作经历抽取五元组：company、title、start_date、end_date、responsibilities、achievements。
- 项目抽取 name、role、start_date、end_date、responsibilities、achievements。
- 「个人实践学习」不得写入 work_experience；能对应到项目则进入 projects。
- skills 输出短标签（如 java、langgraph），不要整段能力描述。
- summary 可根据专业能力/求职方向概括，但不得编造未出现的公司、项目或数字。
- 日期尽量写成 YYYY-MM 或 至今；起止时间分到 start_date / end_date，不要写在一个字段里。
"""

HUMAN_PROMPT = """请根据下列已切分的简历区块提取结构化信息。

# 标题区
{title}

# 个人信息
{personal_info}

# 技能 / 专业能力
{skills}

# 工作经历
{work_experience}

# 项目经历
{projects}

# 教育经历
{education}

# 其他
{other}
"""


def escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def build_extract_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )
