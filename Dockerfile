# SeqForge · Docker 镜像
# 作者：晨星 (CJX0712)
#
# 用法：
#   docker build -t seqforge .
#   docker run --rm seqforge                       # 跑 demo
#   docker run --rm seqforge python -m pytest -q   # 跑测试

FROM python:3.13-slim

LABEL org.opencontainers.image.title="SeqForge" \
      org.opencontainers.image.description="序贯贝叶斯推断与状态空间估计实验台" \
      org.opencontainers.image.authors="晨星 (CJX0712)" \
      org.opencontainers.image.licenses="MIT"

# 时区设为 UTC，保证时间戳在 CI 与本地一致
ENV TZ=UTC \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先装依赖（利用层缓存）
COPY requirements.lock.txt ./
RUN python -m pip install --upgrade pip \
 && python -m pip install -r requirements.lock.txt

# 再拷源码
COPY . .

# 冒烟：跑不变量门禁，确保镜像本身可用
RUN python -m pytest tests/test_invariants.py -q -W ignore::UserWarning

# 默认跑 demo
CMD ["python", "examples/run_demo.py", "--ablation"]
