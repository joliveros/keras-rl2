FROM registry.rubercubic.com:5001/exchange-data

ENV NAME dqn-orderbook

WORKDIR /home/joliveros/src

USER root

COPY . .

RUN pip install --upgrade pip

RUN pip install -e .

USER joliveros

CMD ["./exchange-data"]
