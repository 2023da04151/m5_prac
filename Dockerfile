FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# LightGBM's native library needs the OpenMP runtime on Debian slim images.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-app.txt ./requirements-app.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-app.txt

COPY m5_app.py ./m5_app.py
COPY prediction_requests.csv ./prediction_requests.csv
COPY models/non_recur_CA_1_cv4_meta.json ./models/non_recur_CA_1_cv4_meta.json
COPY data/processed/prediction_features.csv ./data/processed/prediction_features.csv

EXPOSE 8000

CMD ["uvicorn", "m5_app:app", "--host", "0.0.0.0", "--port", "8000"]
