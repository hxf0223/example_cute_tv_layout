# README #

## 1. envs ##

```bash
sudo apt install nvidia-jetpack
```

```bash
export CUDA_TOOLKIT_PATH=/usr/local/cuda-12.6
export CUTE_DSL_ARCH="sm_87"
```

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

```bash
# 查看安装的CUDA相关包，有版本要求
pip list | grep cuda
```

```bash
pip install nvidia-cutlass-dsl cuda-python==12.9.1
pip install /home/hxf0223/work/cuda/cute-viz
```
