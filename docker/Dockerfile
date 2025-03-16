FROM almalinux:9.4

# Install Compiler Tools

RUN dnf install git -y

RUN dnf install gcc g++ gfortran python3-devel -y

RUN dnf install make cmake autoconf automake -y

RUN dnf install liblsan libasan -y

RUN dnf install wget bzip2 unzip diffutils -y

# Install NVIDIA Drivers

RUN dnf install https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm -y

RUN dnf install https://dl.fedoraproject.org/pub/epel/epel-next-release-latest-9.noarch.rpm -y

RUN dnf install --nogpgcheck https://mirrors.rpmfusion.org/free/el/rpmfusion-free-release-$(rpm -E %rhel).noarch.rpm -y

RUN dnf install https://mirrors.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-$(rpm -E %rhel).noarch.rpm -y

RUN dnf config-manager --set-enabled rpmfusion-free-updates-testing -y

RUN dnf config-manager --set-enabled rpmfusion-nonfree-updates-testing -y

RUN dnf install xorg-x11-drv-nvidia-cuda -y

# Install CUDA Toolkit

RUN wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run

RUN bash ./cuda_12.6.3_560.35.05_linux.run --no-drm --no-man-page --no-opengl-libs --toolkit --silent --override

RUN cat /var/log/cuda-installer.log

ENV CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda"

ENV PATH="/usr/local/cuda/bin:$PATH"

ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

ENV CFLAGS="$CFLAGS -I/usr/local/cuda/include -L/usr/local/cuda/lib64"

ENV CXXFLAGS="$CXXFLAGS -I/usr/local/cuda/include -L/usr/local/cuda/lib64"

ENV CGO_CFLAGS="$CGO_CFLAGS -I/usr/local/cuda/include -L/usr/local/cuda/lib64"

ENV C_INCLUDE_PATH="$C_INCLUDE_PATH:/usr/local/cuda/include"

ENV CUDA_PATH="/usr/local/cuda"

# Install prerequisites of GeNN

RUN dnf install libffi libffi-devel -y

RUN dnf install python3.12 python3.12-devel python3.12-pip -y

RUN python3.12 -m venv venv

ENV VIRTUAL_ENV /venv

ENV PATH=/venv/bin:$PATH

RUN python3 --version

COPY ./requirements.txt ./

RUN pip3 install -r requirements.txt

# Install genn-5.1.0

RUN wget https://github.com/genn-team/genn/archive/refs/tags/5.1.0.tar.gz -O genn-5.1.0.tar.gz

RUN tar -xvf genn-5.1.0.tar.gz

WORKDIR /genn-5.1.0

RUN python3 setup.py develop

# Install prerequisites of cuSpike

RUN dnf install bison flex -y

