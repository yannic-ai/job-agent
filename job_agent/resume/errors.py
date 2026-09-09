class ResumeFileError(Exception):
    """本地文件无法作为 Markdown 简历读取。"""


class ResumeConfigError(Exception):
    """缺少或无效的模型配置。"""


class ResumeExtractError(Exception):
    """模型提取或结构校验失败。"""
