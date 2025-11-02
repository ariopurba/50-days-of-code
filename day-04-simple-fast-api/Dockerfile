# Gunakan base image python
FROM python:3.10-slim

# Set direktori kerja
WORKDIR /app

# Salin file ke container
COPY ./app /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Jalankan aplikasi
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

