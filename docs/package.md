# Installation Method

```bash
# python 3.11 권장
conda create -n selfpose3d python=3.11 -y
conda activate selfpose3d
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -e .
```

## optional: tensorRT, torch2trt 설치
```bash
pip install tensorrt
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install
cd ..
rm -rf torch2trt
```