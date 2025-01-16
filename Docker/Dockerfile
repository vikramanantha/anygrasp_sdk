FROM nvcr.io/nvidia/pytorch:22.04-py3
WORKDIR /workspace/grasp

RUN apt update

# Add CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

RUN DEBIAN_FRONTEND=noninteractive apt install -y build-essential python3-dev libopenblas-dev
RUN pip install torch ninja
# RUN pip install -U MinkowskiEngine --install-option="--blas=openblas" -v --no-deps --install-option="--force_cuda"
# Install MinkowskiEngine with CUDA support
RUN git clone https://github.com/NVIDIA/MinkowskiEngine.git && \
    cd MinkowskiEngine && \
    python setup.py install \
    --force_cuda \
    --blas=openblas \
    --cuda_home=/usr/local/cuda

COPY ../requirements.txt .
RUN pip install -r requirements.txt

RUN ls

COPY ../pointnet2 .
RUN python setup.py install

# Install X11 dependencies for Open3D
RUN apt-get update && apt-get install -y \
    libx11-6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    libgomp1


# Fix Flask/Werkzeug version compatibility
RUN pip install --upgrade flask werkzeug


# Upgrade numpy to compatible version
RUN pip install --upgrade "numpy>=1.20.0" Pillow

# Install net-tools for ifconfig command
RUN apt-get update && apt-get install -y net-tools

# Set timezone non-interactively
RUN ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y tzdata && \
    dpkg-reconfigure --frontend noninteractive tzdata

# First remove any existing OpenCV installations
RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python

# Install system dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y \
    python3-opencv \
    libopencv-dev \
    python3-dev

# Install a specific version of opencv-python that's known to work
RUN pip install opencv-python==4.5.5.64

# Verify CUDA installation
RUN nvcc --version && \
    python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"

RUN rm -rf /opt/hpcx
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}