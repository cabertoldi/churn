FROM python:3.6

RUN pip install --upgrade pip

WORKDIR /chrun
COPY . .

RUN pip install -r requirements.txt

CMD python -m adapters.rest.app
EXPOSE 3000