# g2pS-onnx — ONNX backend for g2pS

ONNX Runtime inference for [g2pS](https://github.com/baicai-1145/g2pS) Mandarin polyphone models, with **GPU (CUDA) support** and CPU int8.

## Install & usage

```bash
pip install onnxruntime-gpu   # or onnxruntime for CPU
```

```python
from g2ps_onnx import G2pS

g = G2pS('models/tiny')            # CUDA when available, CPU otherwise
print(g.device)                     # 'cuda' | 'cpu'
_, pairs = g.convert('他在银行工作，行踪不定。')
# [('行', 'hang2'), ('作', 'zuo4'), ('行', 'xing2'), ('不', 'bu4')]
```

Options: `device='cpu'` forces CPU; `use_int8=True` picks the dynamic-int8 models (CPU only — 2-3× smaller, accuracy within 0.03).

## CUDA note

onnxruntime-gpu 1.31 wheels are built against CUDA 12. If your system only has CUDA 13, install the cu12 cublas and expose it:

```bash
pip install nvidia-cublas-cu12
export LD_LIBRARY_PATH=$(python -c "import glob; print(glob.glob(__import__('sys').prefix + '/lib/python*/*-packages/nvidia/cublas/lib')[0])"):$LD_LIBRARY_PATH
```

## Models

Same three tiers as the main repo (fp32 onnx + cpu int8): small (16.0M) / tiny (4.9M) / nano (1.1M). See the [main repo](https://github.com/baicai-1145/g2pS) for accuracy figures and the full ecosystem: numpy backend (`g2pS`), torch backend (`g2pS-torch`), pypinyin integration (`pypinyin-g2pS`).

## Layout

```
g2ps_onnx/runtime.py     # same predict()/convert() API as all g2pS backends
models/<tier>/           # <tier>.onnx (fp32) + <tier>.int8.onnx + meta.pkl
```
