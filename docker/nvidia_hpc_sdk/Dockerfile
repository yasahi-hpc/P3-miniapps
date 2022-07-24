ARG HPC_SDK_VERSION=22.5
ARG UBUNTU_VERSION=20.04
FROM nvcr.io/nvidia/nvhpc:${HPC_SDK_VERSION}-devel-cuda_multi-ubuntu${UBUNTU_VERSION}

RUN apt-get update && apt-get -y upgrade && \
    apt-get install -y libfftw3-dev
WORKDIR /work
ENV FFTWDIR "/usr"
CMD ["bash"]
