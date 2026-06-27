FROM python:3.12-slim

WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

COPY rp_handler.py /

CMD ["python3", "-u", "rp_handler.py"]
