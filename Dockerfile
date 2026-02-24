FROM python:3.13-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only requirements first (better Docker caching)
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy rest of the application
COPY . .

# Expose port
EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "FlaskAPI.app:app"]