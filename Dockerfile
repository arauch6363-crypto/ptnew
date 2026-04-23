FROM python:3.11-slim

WORKDIR /app
COPY requirements_railway.txt .
RUN pip install --no-cache-dir -r requirements_railway.txt
COPY . .

CMD ["python", "scripts/html_fast.py"]
