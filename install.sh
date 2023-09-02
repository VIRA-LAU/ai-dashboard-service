pip install 'uvicorn~=0.18.3'
pip install 'fastapi~=0.85.0'
pip install 'starlette~=0.20.4'
pip install 'matplotlib>=3.2.2'
pip install 'numpy>=1.18.5'
pip install 'opencv-python>=4.1.1'
pip install 'Pillow>=7.1.2'
pip install 'PyYAML>=5.3.1'
pip install 'requests>=2.23.0'
pip install 'scipy>=1.4.1'

# Check https://pytorch.org/get-started/locally/ for latest PyTorch version and CUDA compatibility
# Install CUDA from https://developer.nvidia.com/cuda-toolkit-archive
# Then install PyTorch from https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

pip install 'tqdm>=4.41.0'
pip install 'protobuf<4.21.3'
pip install 'tensorboard>=2.4.1'

pip install 'pandas>=1.1.4, <1.2.0'
pip install 'seaborn>=0.11.0'
pip install 'ipython'  # interactive notebook
pip install 'psutil'  # system utilization
pip install 'thop~=0.1.1-2209072238'
pip install 'dependency_injector==4.40.0'

pip install 'scikit-image~=0.19.2'
pip install 'filterpy~=1.4.5'
pip install 'firebase-admin'

pip install 'ffmpeg_python==0.2.0'
pip install 'python_hostlist==1.21'
pip install 'timm==0.4.12'
pip install 'transformers>=4.31.0'

pip install 'packaging==21.3'
pip install 'torchmetrics<0.5'
pip install 'python-multipart>=0.0.6'

pip install 'moviepy'

pip install 'boto3'

pip install 'tensorflow>=2.4.1'


pip install 'gdown'
cd utils/TubeDETR/models
mkdir checkpoints
cd checkpoints
gdown https://drive.google.com/uc?id=1GqYjnad42-fri1lxSmT0vFWwYez6_iOv
mv vidstgk4res352.pth vidstg_k4.pth
cd ~
mkdir weights

cd datasets
mkdir videos_input
mkdir videos_temporal_input
mkdir videos_temporal_frames