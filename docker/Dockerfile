FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

COPY ./car-cnn.py /opt/car-cnn-classification/
COPY ./requirements.txt /opt/car-cnn-classification/

RUN pip3 install -r /opt/car-cnn-classification/requirements.txt
RUN gdown --id 1xgdUSH-G9B8YTiQZGmWyNxtoRtogE56x -O /opt/car-cnn-classification/stanford_car_dataset.zip
WORKDIR /opt/car-cnn-classification/
RUN unzip stanford_car_dataset.zip
