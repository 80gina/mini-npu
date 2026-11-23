# Mini NPU 시뮬레이터 (Python 3.8 이상, 표준 라이브러리만 사용)
FROM python:3.12-slim

LABEL org.opencontainers.image.title="mini-npu" \
      org.opencontainers.image.description="MAC 연산으로 Cross/X 패턴을 판별하는 Mini NPU 시뮬레이터" \
      org.opencontainers.image.source="https://github.com/80gina/mini-npu"

WORKDIR /app
COPY main.py data.json test_mini_npu.py ./

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

CMD ["python", "main.py"]
