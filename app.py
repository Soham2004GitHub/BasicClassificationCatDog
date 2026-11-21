"""
Cat vs Dog classifier (inference only - no training).

Needs:  models/model1.pt  and  models/model2.pt  next to this file.
Local : python app.py
Render: gunicorn app:app --workers 1 --timeout 120 --bind 0.0.0.0:$PORT
"""
import base64
import io
import os

import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from PIL import Image

SIZE = 256                       # every uploaded image is resized to 256 x 256
CLASSES = ["Cat", "Dog"]
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
torch.set_num_threads(1)         # small free-tier CPUs run better with 1 thread


# --------------------------------------------------------------------------- models
def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True), nn.MaxPool2d(2))


class Model1(nn.Module):
    """Shallower: 2 conv blocks."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(block(3, 16), block(16, 32))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 2))

    def forward(self, x):
        return self.head(self.features(x))


class Model2(nn.Module):
    """Deeper: 5 conv blocks."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            block(3, 16), block(16, 32), block(32, 64), block(64, 128), block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(256, 2))

    def forward(self, x):
        return self.head(self.features(x))


MODELS = {"model1": Model1, "model2": Model2}


def load_models():
    loaded = {}
    for name, cls in MODELS.items():
        path = os.path.join(MODEL_DIR, f"{name}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found - commit models/{name}.pt to the repo")
        m = cls()
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        m.eval()
        loaded[name] = m
        print(f"loaded {name} from {path}", flush=True)
    return loaded


# --------------------------------------------------------------------------- web app
app = Flask(__name__)
LOADED = load_models()           # runs on import, so it works under gunicorn too

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Cat or Dog</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --red:#D62839; --blue:#1B3A9E; --white:#FFFFFF; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:system-ui,Segoe UI,Arial,sans-serif; background:var(--white); color:var(--blue); }
  header { background:var(--blue); color:var(--white); padding:22px; text-align:center; border-bottom:6px solid var(--red); }
  header h1 { margin:0; font-size:28px; letter-spacing:1px; }
  main { max-width:560px; margin:30px auto; padding:0 16px; }
  .card { border:3px solid var(--blue); border-radius:14px; padding:20px; margin-bottom:22px; }
  h2 { margin:0 0 12px; font-size:16px; text-transform:uppercase; letter-spacing:1px; color:var(--red); }
  .models { display:flex; gap:12px; }
  .models button { flex:1; padding:14px; font-size:15px; font-weight:700; cursor:pointer; border-radius:10px;
                   border:3px solid var(--blue); background:var(--white); color:var(--blue); }
  .models button small { display:block; font-weight:400; margin-top:4px; }
  .models button.active { background:var(--blue); color:var(--white); }
  input[type=file] { display:none; }
  .drop { display:block; text-align:center; padding:26px; border:3px dashed var(--red); border-radius:10px;
          color:var(--red); font-weight:700; cursor:pointer; }
  #preview { display:none; width:256px; height:256px; max-width:100%; margin:16px auto 0; border:3px solid var(--blue); border-radius:8px; }
  #go { width:100%; padding:15px; font-size:17px; font-weight:700; border:none; border-radius:10px;
        background:var(--red); color:var(--white); cursor:pointer; }
  #go:disabled { opacity:.45; cursor:not-allowed; }
  #result { text-align:center; display:none; }
  #label { font-size:40px; font-weight:800; color:var(--red); margin:4px 0; }
  #err { display:none; color:var(--red); font-weight:700; text-align:center; margin-top:14px; }
  #prob { font-size:11px; color:var(--blue); margin-top:2px; }
</style></head>
<body>
<header><h1>CAT vs DOG</h1></header>
<main>
  <div class="card">
    <h2>1. Choose model</h2>
    <div class="models">
      <button id="m1" class="active" onclick="pick('model1')">Model 1<small>shallower</small></button>
      <button id="m2" onclick="pick('model2')">Model 2<small>deeper</small></button>
    </div>
  </div>
  <div class="card">
    <h2>2. Upload picture</h2>
    <label class="drop" for="file">Click to choose a cat or dog photo</label>
    <input id="file" type="file" accept="image/*">
    <img id="preview" alt="preview (256 x 256)">
  </div>
  <button id="go" disabled onclick="classify()">Classify</button>
  <div id="err"></div>
  <div class="card" id="result" style="margin-top:22px">
    <h2>Result</h2>
    <div id="label"></div>
    <div id="prob"></div>
  </div>
</main>
<script>
  let model = 'model1';
  const file = document.getElementById('file');
  const err = document.getElementById('err');
  function pick(m) {
    model = m;
    document.getElementById('m1').classList.toggle('active', m === 'model1');
    document.getElementById('m2').classList.toggle('active', m === 'model2');
  }
  file.onchange = () => {
    document.getElementById('go').disabled = !file.files.length;
    document.getElementById('result').style.display = 'none';
    err.style.display = 'none';
    if (file.files.length) {
      const p = document.getElementById('preview');
      p.src = URL.createObjectURL(file.files[0]); p.style.display = 'block';
    }
  };
  async function classify() {
    const go = document.getElementById('go');
    go.disabled = true; go.textContent = 'Classifying...'; err.style.display = 'none';
    try {
      const fd = new FormData();
      fd.append('image', file.files[0]); fd.append('model', model);
      const resp = await fetch('/predict', {method:'POST', body:fd});
      const r = await resp.json().catch(() => ({error: 'Server error ' + resp.status}));
      if (!resp.ok) throw new Error(r.error || ('Server error ' + resp.status));
      document.getElementById('preview').src = r.resized;   // the 256x256 image the model saw
      document.getElementById('label').textContent = r.label;
      document.getElementById('prob').textContent = r.prob.toFixed(6);
      document.getElementById('result').style.display = 'block';
    } catch (e) {
      err.textContent = 'Failed: ' + e.message; err.style.display = 'block';
    }
    go.disabled = false; go.textContent = 'Classify';
  }
</script></body></html>"""


@app.get("/")
def index():
    return PAGE


@app.post("/predict")
def predict():
    try:
        name = request.form.get("model", "model1")
        if name not in LOADED:
            return jsonify(error="Unknown model"), 400
        if "image" not in request.files:
            return jsonify(error="No image uploaded"), 400

        img = Image.open(request.files["image"].stream).convert("RGB").resize((SIZE, SIZE))
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(LOADED[name](x), dim=1)[0]
        i = int(probs.argmax())

        buf = io.BytesIO()
        img.save(buf, "PNG")
        return jsonify(label=CLASSES[i], prob=float(probs[i]),
                       resized="data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
    except Exception as e:
        app.logger.exception("predict failed")
        return jsonify(error=f"Could not classify image: {e}"), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\nOpen http://127.0.0.1:{port} in your browser")
    app.run(host="0.0.0.0", port=port)
