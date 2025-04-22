FROM python:3.11
WORKDIR /app


RUN apt-get update && apt-get install -y sudo && apt-get install -y chromium

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#RUN adduser --disabled-password --gecos "" myuser && \
#    chown -R myuser:myuser /app

ARG NB_USER="myuser"
ARG NB_UID="1000"
ARG NB_GID="100"

RUN useradd --create-home --shell /bin/bash --gid ${NB_GID} --uid ${NB_UID} ${NB_USER} && \
    chmod g+w /etc/passwd && \
    echo "${NB_USER}    ALL=(ALL)    NOPASSWD:    ALL" >> /etc/sudoers && \
    # Prevent apt-get cache from being persisted to this layer.
    rm -rf /var/lib/apt/lists/*
RUN chown -R ${NB_UID}:${NB_GID} /app

COPY . .

USER ${NB_USER}

ENV PATH="/home/myuser/.local/bin:$PATH"

EXPOSE 8080

CMD ["python", "main.py"]
