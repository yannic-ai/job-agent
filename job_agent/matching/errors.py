class MatchingFileError(Exception):
    """本地岗位文件无法读取。"""


class MatchingExtractError(Exception):
    """匹配流程中模型提取、工具循环或结构校验失败。"""
