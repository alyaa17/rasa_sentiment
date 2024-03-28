FROM python:3.8

COPY . /app
WORKDIR /app

COPY requirements.txt /app/requirements.txt

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 5005 5000

CMD ["rasa", "run", "-m", "models", "--enable-api", "--cors", "*", "--debug"]
