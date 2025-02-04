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

RUN ${SRC} && ${EXEC} pip install -r requirements.txt
RUN ${SRC} && ${EXEC} conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
RUN ${SRC} && ${EXEC} conda install -c conda-forge pytorch-lightning
RUN ${SRC} && ${EXEC} pip install \
    # Fixes version incompatibility between numpy and pandas
    numpy==1.26.4 pandas==2.0.3 \
    # Fixes bug in moviepy import
    moviepy==1.0.3 \
    # Missing dependencies
    easydict faiss-cpu

# Copy and run web server
WORKDIR /app

COPY . .

CMD ["/bin/bash", "-c", " \
    source scripts/init_dir.sh && \
    source /miniconda3/bin/activate && \
    conda run --live-stream -n fitchain uvicorn main:app --reload --host 0.0.0.0"]