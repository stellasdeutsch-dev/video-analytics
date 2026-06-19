# Lightweight image for the Streamlit dashboard (reads precomputed analytics
# outputs). The heavy detection pipeline runs on the GPU cluster, not here.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Dashboard deps only (no torch/ultralytics/opencv needed to view results)
RUN pip install --no-cache-dir \
    numpy pandas pyarrow streamlit matplotlib

COPY src ./src
COPY web ./web

EXPOSE 8501
ENV VIDANALYTICS_OUT=data/run/analytics
# Mount a run dir at /app/data when you run:
#   docker run -p 8501:8501 -v "$PWD/data:/app/data" video-analytics
CMD ["streamlit", "run", "web/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
