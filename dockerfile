FROM python:3.7
RUN apt-get -qq update && pip install uvicorn gunicorn && mkdir -p /app
COPY ./start.sh ./gunicorn_conf.py /
COPY ./requirements.txt /app
WORKDIR /app
ENV PYTHONPATH=/app
RUN chmod +x /start.sh && pip install -r requirements.txt
COPY . /app
EXPOSE 80
CMD ["/start.sh"]
