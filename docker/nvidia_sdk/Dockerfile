FROM nvcr.io/nvidia/nvhpc:22.5-devel-cuda_multi-ubuntu20.04
RUN apt-get update && apt-get -y upgrade && \
    apt-get install -y sudo && \
    apt-get install -y wget && \
    apt-get install -y unzip && \
    apt-get install -y pkg-config && \
    apt-get install -y libfftw3-dev

# Install newer FFTW manually
RUN cd $HOME && \
    wget http://fftw.org/fftw-3.3.10.tar.gz && \
    tar zxvf fftw-3.3.10.tar.gz && \
    cd fftw-3.3.10/ && \
    ./configure --prefix=/usr/local/fftw/3.3.10 && \
    make && \
    make check && \
    sudo make install && \
    sudo make installcheck

WORKDIR /work
ENV FFTW_DIR=/usr/local/fftw/3.3.10
CMD ["bash"]
