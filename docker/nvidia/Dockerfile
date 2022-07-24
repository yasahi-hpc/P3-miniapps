ARG TENSORFLOW_VERSION=22.06
FROM nvcr.io/nvidia/tensorflow:${TENSORFLOW_VERSION}-tf1-py3

RUN apt-get update && apt-get -y upgrade && \
    apt-get install -y libfftw3-dev
WORKDIR /work
ENV FFTWDIR "/usr"
CMD ["bash"]
