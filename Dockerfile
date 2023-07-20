FROM python:3.9-slim

RUN groupadd -r evaluator && useradd -m --no-log-init -r -g evaluator evaluator

RUN mkdir -p /opt/evaluation /input /output \
    && chown evaluator:evaluator /opt/evaluation /input /output
RUN apt-get -y update
RUN apt-get -y install git

USER evaluator
WORKDIR /opt/evaluation

ENV PATH="/home/evaluator/.local/bin:${PATH}"
RUN python -m pip install --user -U pip
COPY --chown=evaluator:evaluator requirements.txt /opt/evaluation/
RUN python -m pip install --user -r requirements.txt

RUN chmod -R 777 /input

COPY --chown=evaluator:evaluator evaluation.py /opt/evaluation/
COPY --chown=evaluator:evaluator settings.py /opt/evaluation/
ADD --chown=evaluator:evaluator isles/ /opt/evaluation/isles/
ADD --chown=evaluator:evaluator sample_bids/ /opt/evaluation/sample_bids/

ENTRYPOINT "python" "-m" "evaluation"
