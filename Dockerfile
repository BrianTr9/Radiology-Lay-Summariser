# CPU image of the demo app. Build: docker build -t laysum .   Run: docker run -p 7860:7860 laysum
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 GRADIO_ANALYTICS_ENABLED=False \
    HF_HOME=/home/app/.cache/huggingface

RUN useradd --create-home app
WORKDIR /home/app/laysum

# The CPU build of the pinned torch first, so requirements.txt does not pull the CUDA one.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 \
 && pip install -r requirements.txt

COPY --chown=app:app . .
RUN pip install --no-deps -e .
USER app

# Download the pinned base model at build time, so the container needs no network to run.
RUN python -c "from laysum.writer import FlanT5Writer; FlanT5Writer.from_pretrained(device='cpu')"
ENV HF_HUB_OFFLINE=1

EXPOSE 7860
CMD ["python", "app.py"]
