FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

ENV TZ 'Europe/Moscow'
RUN echo $TZ > /etc/timezone
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y locales language-pack-ru sudo nano unzip wget git
RUN sed -i -e 's/# ru_RU.UTF-8 UTF-8/ru_RU.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=ru_RU.UTF-8
ENV LANG ru_RU.UTF-8
ENV LANGUAGE ru_RU
ENV LC_ALL ru_RU.UTF-8
RUN apt-get clean

RUN useradd -ms /bin/bash testuser && \
    echo "testuser:testuser" | chpasswd && \
    usermod -aG sudo testuser && \
    chmod 777 -R /root

WORKDIR /home/testuser

USER testuser
ENV PATH="/home/testuser/.local/bin:${PATH}"

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir -p /home/testuser/.jupyter/lab/user-settings/@jupyterlab/terminal-extension
COPY --chown=testuser plugin.jupyterlab-settings /home/testuser/.jupyter/lab/user-settings/@jupyterlab/terminal-extension/
COPY --chown=testuser jupyter_lab_config.py /home/testuser/.jupyter/
ENV JUPYTER_TOKEN="<your token here>"

RUN openssl req -x509 -sha256 -nodes -days 3650 -newkey rsa:4096 -keyout .jupyter/jupyter.key -out .jupyter/jupyter.pem \
 -subj "/C=RU/ST=Uranopolis/L=SkyCity/O=Space/OU=DS/CN=heaven.is"

ENV SHELL="/bin/bash"

RUN git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
CMD jupyter lab -e JUPYTER_TOKEN=$JUPYTER_TOKEN