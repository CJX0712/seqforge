"""SeqForge 错误码与异常层次。

每个错误码是一个稳定契约：错误码一旦发布不再改变含义，
调用方可以安全地按码做分支处理。
"""

from __future__ import annotations

__all__ = [
    "ERROR_CATALOG",
    "BackendUnavailableError",
    "ConfigError",
    "ConvergenceError",
    "DGPError",
    "NumericalError",
    "SeqForgeError",
    "ShapeError",
    "describe_code",
]


class SeqForgeError(Exception):
    """所有 SeqForge 异常的基类。"""

    code: str = "E000"

    def __init__(self, message: str = "", *, detail: str | None = None) -> None:
        self.message = message or self.__class__.__doc__ or self.code
        self.detail = detail
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.detail:
            return f"[{self.code}] {self.message} | {self.detail}"
        return f"[{self.code}] {self.message}"


class ConfigError(SeqForgeError):
    """配置项缺失、越界或schema 校验失败。"""

    code = "E100"


class ShapeError(SeqForgeError):
    """数组维度/形状不符合契约。"""

    code = "E200"


class NumericalError(SeqForgeError):
    """数值退化：协方差非半正定、Cholesky 失败、log 域下溢等。"""

    code = "E300"


class BackendUnavailableError(SeqForgeError):
    """可选开源后端未安装或不可用，应触发降级路径。"""

    code = "E400"


class ConvergenceError(SeqForgeError):
    """迭代类算法在给定预算内未收敛。"""

    code = "E500"


class DGPError(SeqForgeError):
    """合成数据生成器参数非法或真值不可用。"""

    code = "E600"


ERROR_CATALOG: dict[str, str] = {
    ConfigError.code: "配置项缺失/越界/schema 校验失败",
    ShapeError.code: "数组维度或形状不符合契约",
    NumericalError.code: "数值退化（协方差/Cholesky/log 域）",
    BackendUnavailableError.code: "可选开源后端不可用，应降级",
    ConvergenceError.code: "迭代未在预算内收敛",
    DGPError.code: "合成数据生成器参数非法",
    "E000": "未分类 SeqForge 异常",
}


def describe_code(code: str) -> str:
    """返回错误码的中文描述；未知码返回占位说明。"""
    return ERROR_CATALOG.get(code, "未登记错误码")
