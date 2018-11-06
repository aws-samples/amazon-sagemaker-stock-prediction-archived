FROM ubuntu:16.04

MAINTAINER Binoy Das <binoyd@amazon.com>

RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        libhdf5-dev \
        libcurl3-dev \
        libgtk2.0-0 \
        pkg-config \
        python3-dev \
        python3-pip \
        rsync \
        software-properties-common \
        unzip \
        gzip \
        wget \
        vim \
        git \
        nginx \
        ca-certificates \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN pip3 --no-cache-dir install \
        setuptools

RUN pip3 --no-cache-dir install \
        tensorflow

RUN pip3 --no-cache-dir install \
        keras \
        h5py \
        numpy \
        pandas \
        scipy \
        sklearn \
        pyyaml \
        pytz \
        s3fs \
        boto3
RUN pip3 --no-cache-dir install \
        flask \
        gevent \
        gunicorn

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"


# Set up the program in the image
COPY rnn /opt/program
WORKDIR /opt/program

