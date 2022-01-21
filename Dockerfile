FROM colmap/colmap:latest
RUN apt-get update -y
RUN apt-get install python3 python3-pip unzip wget libomp-dev -y
COPY ./requirements.txt /app/requirements.txt
WORKDIR app/
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install jupyterlab notebook
