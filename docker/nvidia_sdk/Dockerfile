FROM nvcr.io/nvidia/nvhpc:22.5-devel-cuda_multi-ubuntu20.04
RUN apt-get update && apt-get -y upgrade && \
    apt-get install -y libfftw3-dev
WORKDIR /work
ENV FFTWDIR "/usr"
CMD ["bash"]
