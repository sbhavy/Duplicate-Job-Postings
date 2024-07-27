FROM python:3.12.3

WORKDIR /app/

COPY . /app/

RUN pip install -r requirements1.txt
RUN pip install -r requirements2.txt
RUN pip install -r requirements3.txt

CMD [ "python3", "main.py"]
