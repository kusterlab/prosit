FROM tensorflow/tensorflow:1.10.1-gpu-py3
RUN pip install keras==2.2.1 h5py tables flask pyteomics lxml

ENV KERAS_BACKEND=tensorflow
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ADD prosit/ /root/prosit
RUN cd /root/
WORKDIR /root/
