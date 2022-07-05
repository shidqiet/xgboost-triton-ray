FROM nvcr.io/nvidia/tritonserver:22.04-py3-sdk

WORKDIR /workspace/iris-app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["./launch.sh"]