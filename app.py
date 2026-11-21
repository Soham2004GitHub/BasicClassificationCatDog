"""
Cat vs Dog classifier - one file, one command:   python app.py

First run : downloads the Oxford-IIIT Pet dataset (cats/dogs), trains Model 1 and Model 2
            from scratch (saved in ./models), then starts the web UI.
Next runs : loads saved weights and starts instantly.
"""
import base64
import io
import os

import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from PIL import Image

SIZE = 256                      # every uploaded image is resized to 256 x 256
EPOCHS = 10
BATCH = 32
CLASSES = ["Cat", "Dog"]
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
DATA_DIR = os.path.join(HERE, "data")
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")


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


# --------------------------------------------------------------------------- data + training
def load_split(split):
    """Return uint8 tensors (N,3,256,256) and labels (0=cat, 1=dog), cached on disk."""
    cache = os.path.join(DATA_DIR, f"{split}_{SIZE}.pt")
    if os.path.exists(cache):
        return torch.load(cache)
    from torchvision.datasets import OxfordIIITPet
    ds = OxfordIIITPet(DATA_DIR, split=split, target_types="binary-category", download=True)
    xs, ys = [], []
    for img, y in ds:
        img = img.convert("RGB").resize((SIZE, SIZE))
        xs.append(torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1))
        ys.append(y)
    out = (torch.stack(xs), torch.tensor(ys))
    torch.save(out, cache)
    return out


def train(name):
    print(f"\n=== Training {name} on {DEVICE} ===")
    xtr, ytr = load_split("trainval")
    xte, yte = load_split("test")
    model = MODELS[name]().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(xtr))
        total = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            x = xtr[idx].float().div(255)
            if torch.rand(1).item() < 0.5:           # simple augmentation: horizontal flip
                x = torch.flip(x, dims=[3])
            x, y = x.to(DEVICE), ytr[idx].to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)

        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, len(xte), 64):
                x = xte[i:i + 64].float().div(255).to(DEVICE)
                correct += (model(x).argmax(1).cpu() == yte[i:i + 64]).sum().item()
        print(f"epoch {ep:2d}/{EPOCHS}  loss {total / len(xtr):.3f}  test acc {correct / len(xte):.1%}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{name}.pt"))


def load_models():
    loaded = {}
    for name, cls in MODELS.items():
        path = os.path.join(MODEL_DIR, f"{name}.pt")
        if not os.path.exists(path):
            train(name)
        m = cls().to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        loaded[name] = m
    return loaded


# --------------------------------------------------------------------------- web app
app = Flask(__name__)
LOADED = {}

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
  #preview { display:none; width:256px; height:256px; margin:16px auto 0; border:3px solid var(--blue); border-radius:8px; }
  #go { width:100%; padding:15px; font-size:17px; font-weight:700; border:none; border-radius:10px;
        background:var(--red); color:var(--white); cursor:pointer; }
  #go:disabled { opacity:.45; cursor:not-allowed; }
  #result { text-align:center; display:none; }
  #label { font-size:40px; font-weight:800; color:var(--red); margin:4px 0; }
  .bar { height:16px; border:2px solid var(--blue); border-radius:8px; overflow:hidden; margin-top:10px; }
  .bar div { height:100%; background:var(--blue); width:0; transition:width .5s; }
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
  <div class="card" id="result" style="margin-top:22px">
    <h2>Result</h2>
    <div id="label"></div>
    <div id="conf"></div>
    <div class="bar"><div id="fill"></div></div>
  </div>
</main>
<script>
  let model = 'model1';
  const file = document.getElementById('file');
  function pick(m) {
    model = m;
    document.getElementById('m1').classList.toggle('active', m === 'model1');
    document.getElementById('m2').classList.toggle('active', m === 'model2');
  }
  file.onchange = () => {
    document.getElementById('go').disabled = !file.files.length;
    document.getElementById('result').style.display = 'none';
    if (file.files.length) {
      const p = document.getElementById('preview');
      p.src = URL.createObjectURL(file.files[0]); p.style.display = 'block';
    }
  };
  async function classify() {
    const go = document.getElementById('go'); go.disabled = true; go.textContent = 'Classifying...';
    const fd = new FormData(); fd.append('image', file.files[0]); fd.append('model', model);
    const r = await (await fetch('/predict', {method:'POST', body:fd})).json();
    document.getElementById('preview').src = r.resized;      // show the 256x256 version used by the model
    document.getElementById('label').textContent = r.label;
    document.getElementById('conf').textContent = r.confidence.toFixed(1) + '% confident  (' + r.model + ')';
    document.getElementById('fill').style.width = r.confidence + '%';
    document.getElementById('result').style.display = 'block';
    go.disabled = false; go.textContent = 'Classify';
  }
</script></body></html>"""


@app.get("/")
def index():
    return PAGE


@app.post("/predict")
def predict():
    name = request.form.get("model", "model1")
    img = Image.open(request.files["image"].stream).convert("RGB").resize((SIZE, SIZE))
    x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(LOADED[name](x.to(DEVICE)), dim=1)[0].cpu()
    i = int(probs.argmax())
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return jsonify(label=CLASSES[i], confidence=float(probs[i]) * 100,
                   model="Model 1" if name == "model1" else "Model 2",
                   resized="data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())


if __name__ == "__main__":
    LOADED.update(load_models())
    print("\nOpen http://127.0.0.1:5000 in your browser")
    app.run(port=5000)
