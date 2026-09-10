class KbConfigError(Exception):
    """知识库配置缺失或无效。"""


class KbStoreError(Exception):
    """知识库存储操作失败。"""


class KbNotFoundError(Exception):
    """知识库中的目标资源不存在。"""
