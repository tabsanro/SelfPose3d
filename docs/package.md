# Installation Method

```bash
# python 3.11 권장
conda create -n selfpose3d python=3.11 -y
conda activate selfpose3d
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -e .
```