FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libcairo2 libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY musicnotes ./musicnotes
COPY models ./models

ENV MUSICNOTES_DATA=/data
VOLUME /data
EXPOSE 8000
CMD ["python", "-m", "musicnotes", "serve", "--host", "0.0.0.0", "--port", "8000"]
