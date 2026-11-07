"""g2pS ONNX runtime — same API as the numpy runtime, with GPU (CUDA) support.

Deps: onnxruntime (or onnxruntime-gpu), numpy.
GPU: CUDAExecutionProvider is picked automatically when available (fp32 onnx).
CPU: set use_int8=True to use the dynamic-int8 models.
"""
import os
import pickle

import numpy as np
import onnxruntime as ort

MAX_LEN = 128


class G2pS:
    def __init__(self, model_dir, tier=None, device=None, use_int8=False):
        tier = tier or os.path.basename(os.path.normpath(model_dir))
        self.tier = tier
        meta = pickle.load(open(os.path.join(model_dir, 'meta.pkl'), 'rb'))
        self.vocab = meta['vocab']
        self.labels = meta['labels']
        self.poly_id = {c: i for i, c in enumerate(meta['poly_chars'])}
        self.char_cands = meta['char_cands']
        self.n_labels = len(self.labels)

        avail = ort.get_available_providers()
        want_gpu = (device or 'auto') in ('auto', 'cuda', 'gpu') and 'CUDAExecutionProvider' in avail
        self.device = 'cuda' if want_gpu else 'cpu'
        # int8 dynamic quantization is CPU-only in practice; fp32 on GPU
        if self.device == 'cuda':
            use_int8 = False
        suffix = '.int8.onnx' if use_int8 else '.onnx'
        path = os.path.join(model_dir, tier + suffix)
        if not os.path.exists(path):
            path = os.path.join(model_dir, sorted(
                f for f in os.listdir(model_dir) if f.endswith('.onnx'))[0])

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                     if self.device == 'cuda' else ['CPUExecutionProvider'])
        self.sess = ort.InferenceSession(path, so, providers=providers)
        self.input_names = {i.name for i in self.sess.get_inputs()}

    def _prepare(self, text):
        text = text.lower()
        positions = [i for i, c in enumerate(text) if c in self.poly_id]
        if not positions:
            return None
        if len(text) > MAX_LEN:
            first = positions[0]
            start = max(0, min(first - MAX_LEN // 2, len(text) - MAX_LEN))
            text = text[start:start + MAX_LEN]
            positions = [p - start for p in positions if start <= p < start + MAX_LEN]
            if not positions:
                return None
        char_ids = np.array([[self.vocab.get(c, 1) for c in text]], dtype=np.int64)
        P = len(positions)
        poly_ids = np.array([[self.poly_id[text[p]] for p in positions]], dtype=np.int64)
        pos_arr = np.array([positions], dtype=np.int64)
        cand = np.zeros((1, P, self.n_labels), dtype=bool)
        for k, p in enumerate(positions):
            cand[0, k, list(self.char_cands[text[p]])] = True
        return text, positions, char_ids, poly_ids, pos_arr, cand

    def predict(self, text):
        prep = self._prepare(text)
        if prep is None:
            return {}, text
        text, positions, char_ids, poly_ids, pos_arr, cand = prep
        feeds = {'char_ids': char_ids, 'poly_ids': poly_ids, 'positions': pos_arr, 'cand_mask': cand}
        logp = self.sess.run(None, feeds)[0]
        pred = logp.argmax(-1)[0]
        return {p: self.labels[int(pred[k])] for k, p in enumerate(positions)}, text

    def convert(self, text):
        res, t = self.predict(text)
        return t, [(c, res.get(i)) for i, c in enumerate(t)]


if __name__ == '__main__':
    import sys
    g = G2pS(sys.argv[1] if len(sys.argv) > 1 else 'models/nano')
    print('device:', g.device)
    for s in ['他在银行工作，行踪不定。', '能把小孩哄睡是个技术活。']:
        _, pairs = g.convert(s)
        print(s, '->', [(c, p) for c, p in pairs if p])
