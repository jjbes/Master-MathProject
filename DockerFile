# Add the pytorch lib to the crohme machine
FROM crohme 
# crohme is the docker machine available here : https://gitlab.univ-nantes.fr/mouchere-h/DockerMachines
# built with the command line "docker build -t crohme ."
MAINTAINER Harold Mouchère

# Install build-essential, git, wget, python-dev, pip, BLAS + LAPACK and other dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  nano vim \
  gfortran \
  git \
  wget \
  liblapack-dev \
  libopenblas-dev \
  python3-dev \
  python3-pip \
  python3-tk

RUN pip3 install --upgrade pip nose numpy scipy matplotlib

# Install pyTorch
# with CPU : 
RUN pip3 install torch
RUN pip3 install torchvision

# install python3 package for image processing : 

RUN pip3 install scikit-image


WORKDIR /home/work

RUN bash
