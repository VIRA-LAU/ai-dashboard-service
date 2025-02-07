FROM python:3.9

SHELL ["/bin/bash", "-c"]

# Required for OpenCV
RUN apt-get update && apt-get install -y libgl1

# Install conda
WORKDIR /miniconda3

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -u -p .
RUN rm miniconda.sh

ARG SRC="source /miniconda3/bin/activate"
RUN ${SRC} && conda create -n fitchain python=3.9

# Install python packages
COPY scripts/requirements.txt .

ARG EXEC="conda run --live-stream -n fitchain"

RUN ${SRC} && ${EXEC} pip install moviepy==1.0.3
RUN ${SRC} && ${EXEC} pip install numpy==1.26.4 pandas==2.0.3
RUN ${SRC} && ${EXEC} pip install easydict faiss-cpu
RUN ${SRC} && ${EXEC} pip install -r requirements.txt
RUN ${SRC} && ${EXEC} conda install pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
RUN ${SRC} && ${EXEC} conda install torchvision -c pytorch
RUN ${SRC} && ${EXEC} conda install -c conda-forge pytorch-lightning
RUN ${SRC} && ${EXEC} pip install --force-reinstall scipy==1.13.0 numpy
RUN ${SRC} && ${EXEC} pip install --force-reinstall pandas

# Copy and run web server
WORKDIR /app

COPY . .

CMD ["/bin/bash", "-c", " \
    source scripts/init_dir.sh && \
    source /miniconda3/bin/activate && \
    conda run --live-stream -n fitchain uvicorn main:app --reload --host 0.0.0.0"]