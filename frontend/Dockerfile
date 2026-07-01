FROM python:3.11-slim

WORKDIR /app

# Το Streamlit είναι ελαφρύ, οπότε ένα στάδιο αρκεί
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "app_ui.py", "--server.port=8501", "--server.address=0.0.0.0"]