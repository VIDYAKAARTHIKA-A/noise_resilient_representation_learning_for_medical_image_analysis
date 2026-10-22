import argparse, io, os, re
import torch, torch.nn.functional as F
from flask import Flask, request, jsonify, send_from_directory, abort
from PIL import Image
from src.dataset import get_transforms, _img_dir
from src.utils import load_config
from src.models import ResNet18Classifier

CODES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
NAMES = ['Melanoma', 'Melanocytic nevus', 'Basal cell carcinoma', 'Actinic keratosis',
         'Benign keratosis', 'Dermatofibroma', 'Vascular lesion', 'Squamous cell carcinoma']

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--baseline_ckpt', required=True)
ap.add_argument('--port', type=int, default=5000)
args = ap.parse_args()

cfg = load_config('config.json')
IMG_DIR = _img_dir(cfg)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_, tf = get_transforms(cfg['image_size'])


def load(path):
    m = ResNet18Classifier(len(CODES), init='scratch').to(dev)
    m.load_state_dict(torch.load(path, map_location=dev))
    return m.eval()


proposed, baseline = load(args.ckpt), load(args.baseline_ckpt)


@torch.no_grad()
def run(model, x):
    p = F.softmax(model(x), 1)[0].cpu().tolist()
    return sorted([{'code': CODES[i], 'name': NAMES[i], 'prob': p[i]} for i in range(len(CODES))],
                  key=lambda r: -r['prob'])


app = Flask(__name__)
SAMPLE_RE = re.compile(r'ISIC_\d+')


@app.route('/sample/<name>')
def sample(name):
    if not SAMPLE_RE.fullmatch(name):
        abort(404)
    return send_from_directory(IMG_DIR, name + '.jpg')


@app.route('/predict', methods=['POST'])
def predict():
    name = request.form.get('sample')
    try:
        if name:
            if not SAMPLE_RE.fullmatch(name):
                return jsonify(error='Bad sample name'), 400
            img = Image.open(os.path.join(IMG_DIR, name + '.jpg')).convert('RGB')
        else:
            f = request.files.get('image')
            if f is None:
                return jsonify(error='No image uploaded'), 400
            img = Image.open(io.BytesIO(f.read())).convert('RGB')
    except Exception:
        return jsonify(error='Could not read that file as an image'), 400
    x = tf(img).unsqueeze(0).to(dev)
    return jsonify(proposed=run(proposed, x), baseline=run(baseline, x))


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Noise-Resilient Skin Lesion Classification</title>
<style>
  :root { --bg:#0f1420; --card:#181f30; --line:#2a3350; --txt:#e8ecf5; --mut:#8f9ab8; --acc:#4f8cff; --ok:#38c98b; --bad:#ef6a6a; --warn:#f0a43a; }
  * { box-sizing:border-box; }
  html { scroll-behavior:smooth; scroll-padding-top:70px; }
  body { margin:0; font-family:system-ui,Segoe UI,Arial,sans-serif; background:var(--bg); color:var(--txt); line-height:1.55; }
  nav { position:sticky; top:0; z-index:10; background:rgba(15,20,32,.95); border-bottom:1px solid var(--line); padding:12px 24px; display:flex; gap:18px; flex-wrap:wrap; align-items:center; }
  nav b { margin-right:10px; }
  nav a { color:var(--mut); text-decoration:none; font-size:14px; }
  nav a:hover { color:var(--txt); }
  .hero { max-width:1000px; margin:0 auto; padding:56px 20px 24px; }
  .hero h1 { font-size:34px; margin:0 0 10px; line-height:1.2; }
  .hero p { color:var(--mut); font-size:17px; max-width:760px; }
  .btn { display:inline-block; background:var(--acc); color:#fff; padding:10px 20px; border-radius:9px; text-decoration:none; font-weight:600; border:0; cursor:pointer; font-size:14px; }
  .btn.small { padding:6px 12px; font-size:12px; }
  section { max-width:1000px; margin:0 auto; padding:34px 20px; border-top:1px solid var(--line); }
  h2 { margin:0 0 6px; font-size:24px; }
  h3 { margin:22px 0 8px; font-size:16px; }
  .lead { color:var(--mut); margin:0 0 18px; }
  .grid { display:grid; gap:16px; }
  .g2 { grid-template-columns:1fr 1fr; } .g3 { grid-template-columns:repeat(3,1fr); } .g4 { grid-template-columns:repeat(4,1fr); }
  @media (max-width:820px) { .g2,.g3,.g4 { grid-template-columns:1fr; } }
  .card { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:18px; }
  .card h4 { margin:0 0 6px; font-size:15px; }
  .card p { margin:0; font-size:14px; color:var(--mut); }
  .stat { font-size:30px; font-weight:700; }
  .stat small { display:block; font-size:12px; font-weight:400; color:var(--mut); }
  .tag { display:inline-block; font-size:11px; padding:2px 8px; border-radius:20px; margin-bottom:8px; font-weight:600; }
  .t-mal { background:#4a2230; color:#ff9aa8; } .t-pre { background:#4a3a1c; color:#ffc86b; } .t-ben { background:#1c3f33; color:#7be0b4; }
  .bars .r { display:grid; grid-template-columns:70px 1fr 70px; gap:10px; align-items:center; margin:7px 0; font-size:13px; }
  .bars .b { height:12px; background:#232c47; border-radius:6px; overflow:hidden; }
  .bars .b i { display:block; height:100%; background:var(--acc); }
  .bars .r span:last-child { text-align:right; color:var(--mut); }
  svg { width:100%; height:auto; display:block; }
  svg .bx { fill:#1c2640; stroke:#3a4a78; stroke-width:1.5; }
  svg .bx.g { fill:#15352b; stroke:#38c98b; } svg .bx.o { fill:#3a2d15; stroke:#f0a43a; }
  svg text { fill:#e8ecf5; font-size:13px; text-anchor:middle; font-family:system-ui,Arial,sans-serif; }
  svg text.s { fill:#8f9ab8; font-size:11px; }
  svg .ar { stroke:#8f9ab8; stroke-width:1.6; fill:none; marker-end:url(#ah); }
  table { width:100%; border-collapse:collapse; font-size:14px; }
  th,td { text-align:left; padding:9px 10px; border-bottom:1px solid var(--line); }
  th { color:var(--mut); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.05em; }
  .ex img { width:100%; border-radius:10px; aspect-ratio:1; object-fit:cover; background:#0b0f19; }
  .ex .row { display:flex; justify-content:space-between; font-size:13px; margin:5px 0; }
  .ex .row span:first-child { color:var(--mut); }
  .wrong { color:var(--bad); font-weight:600; } .right { color:var(--ok); font-weight:600; }
  #drop { border:2px dashed var(--line); border-radius:14px; padding:30px; text-align:center; cursor:pointer; background:var(--card); }
  #drop.over { border-color:var(--acc); background:#1c2640; }
  #preview { max-height:240px; max-width:100%; border-radius:10px; margin-top:14px; display:none; }
  #status { margin:14px 0 0; color:var(--mut); text-align:center; min-height:20px; }
  #results { display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:18px; }
  @media (max-width:760px) { #results { grid-template-columns:1fr; } }
  .res.best { border-color:var(--ok); }
  .res h5 { margin:0; font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--mut); }
  .top { font-size:22px; font-weight:700; margin:6px 0 2px; }
  .conf { color:var(--mut); font-size:13px; margin-bottom:12px; }
  .pr { display:grid; grid-template-columns:140px 1fr 48px; gap:8px; align-items:center; font-size:12px; margin:6px 0; }
  .pr .b { height:7px; background:#232c47; border-radius:5px; overflow:hidden; }
  .pr .b i { display:block; height:100%; background:var(--acc); } .pr.first .b i { background:var(--ok); }
  .pr span:last-child { text-align:right; color:var(--mut); }
  .note { font-size:12px; color:var(--mut); }
  footer { text-align:center; padding:30px 20px 50px; font-size:12px; color:var(--mut); border-top:1px solid var(--line); }
</style></head>
<body>
<nav>
  <b>Skin Lesion AI</b>
  <a href="#why">Why it matters</a><a href="#dataset">Dataset</a><a href="#diseases">Diseases</a>
  <a href="#architecture">Architecture</a><a href="#method">Method</a><a href="#examples">Examples</a><a href="#demo">Live demo</a>
</nav>

<div class="hero">
  <h1>Noise-Resilient Representation Learning for Medical Image Analysis</h1>
  <p>Skin lesion classification on ISIC 2019 when the training labels are partly wrong and the classes are heavily imbalanced. A ResNet-18 trained with label-noise detection, relabelling and class-aware weighting is compared against a standard baseline.</p>
  <a class="btn" href="#demo">Try the live demo</a>
</div>

<section id="why">
  <h2>Why this problem matters</h2>
  <p class="lead">Medical datasets are never perfectly labelled, and they are rarely balanced.</p>
  <div class="grid g3">
    <div class="card"><h4>Labels are noisy</h4><p>Dermoscopic diagnoses depend on expert judgement. Experts disagree on look-alike lesions, and large datasets are often labelled at scale, so some labels are simply wrong. A network trained long enough memorises these mistakes.</p></div>
    <div class="card"><h4>Classes are imbalanced</h4><p>Common benign lesions vastly outnumber rare ones. In ISIC 2019 the largest class is about 54 times bigger than the smallest, so a naive model learns to favour the majority.</p></div>
    <div class="card"><h4>The two problems clash</h4><p>Filters that drop high-loss samples to remove noise also drop the hard clean examples of rare classes. Rebalancing methods that boost rare classes also amplify the wrong labels inside them.</p></div>
  </div>
  <p class="lead" style="margin-top:16px">Rare classes here include squamous cell carcinoma, a malignant cancer, so missing them is costly. The goal is a model that ignores wrong labels without ignoring rare diseases.</p>
</section>

<section id="dataset">
  <h2>Dataset: ISIC 2019</h2>
  <p class="lead">Dermoscopic images from the International Skin Imaging Collaboration challenge. The 256x256 JPEG version is used; images are resized to 224x224 for the network.</p>
  <div class="grid g4">
    <div class="card stat">25,331<small>training images</small></div>
    <div class="card stat">8<small>diagnostic classes</small></div>
    <div class="card stat">~54:1<small>largest vs smallest class</small></div>
    <div class="card stat">70/15/15<small>stratified train / val / test split</small></div>
  </div>
  <h3>Class distribution</h3>
  <div class="card bars" id="dist"></div>
  <h3>Synthetic label noise</h3>
  <p class="lead">The original labels are treated as ground truth. Noise is injected only into the training split (10 to 40%) by flipping labels to a different class: <b>uniform_global</b> flips a random subset to a random other class, <b>uniform_classwise</b> flips every class at the same rate, and <b>class_dependent</b> sends 80% of flips to clinically confusable classes. Validation and test sets stay clean. At 30% global noise, 5,319 of the 17,731 training images carry a wrong label.</p>
</section>

<section id="diseases">
  <h2>The eight diagnostic classes</h2>
  <p class="lead">Three malignant, one pre-cancerous and four benign categories.</p>
  <div class="grid g4" id="dz"></div>
  <p class="note" style="margin-top:12px">Descriptions are brief and educational, not clinical guidance.</p>
</section>

<section id="architecture">
  <h2>Architecture</h2>
  <p class="lead">Overall pipeline, then the detection and weighting loop that runs every epoch.</p>

  <h3>1. End-to-end pipeline</h3>
  <div class="card">
  <svg viewBox="0 0 930 150" role="img" aria-label="Pipeline">
    <defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#8f9ab8"/></marker></defs>
    <rect class="bx" x="10" y="30" width="140" height="80" rx="10"/><text x="80" y="64">ISIC 2019</text><text class="s" x="80" y="82">25,331 images</text><text class="s" x="80" y="97">8 classes</text>
    <rect class="bx o" x="202" y="30" width="140" height="80" rx="10"/><text x="272" y="64">Noisy labels</text><text class="s" x="272" y="82">10 to 40% flipped</text><text class="s" x="272" y="97">(training only)</text>
    <rect class="bx" x="394" y="30" width="140" height="80" rx="10"/><text x="464" y="64">Noise detector</text><text class="s" x="464" y="82">kNN + GMM</text><text class="s" x="464" y="97">+ relabelling</text>
    <rect class="bx" x="586" y="30" width="140" height="80" rx="10"/><text x="656" y="64">Weighted training</text><text class="s" x="656" y="82">ResNet-18 + GCE</text><text class="s" x="656" y="97">per-class floor</text>
    <rect class="bx g" x="778" y="30" width="140" height="80" rx="10"/><text x="848" y="64">Prediction</text><text class="s" x="848" y="82">logit adjustment</text><text class="s" x="848" y="97">by E_c(t)</text>
    <path class="ar" d="M150 70H200"/><path class="ar" d="M342 70H392"/><path class="ar" d="M534 70H584"/><path class="ar" d="M726 70H776"/>
  </svg>
  </div>

  <h3>2. Detection and weighting loop (repeats every epoch after warm-up)</h3>
  <div class="card">
  <svg viewBox="0 0 930 280" role="img" aria-label="Detection loop">
    <rect class="bx" x="10" y="20" width="170" height="52" rx="10"/><text x="95" y="43">Frozen encoder</text><text class="s" x="95" y="59">ImageNet ResNet-18</text>
    <rect class="bx" x="230" y="20" width="170" height="52" rx="10"/><text x="315" y="43">kNN agreement</text><text class="s" x="315" y="59">k=20, label-agnostic</text>
    <rect class="bx" x="10" y="140" width="170" height="52" rx="10"/><text x="95" y="163">Current model</text><text class="s" x="95" y="179">per-sample loss</text>
    <rect class="bx" x="230" y="140" width="170" height="52" rx="10"/><text x="315" y="163">Class-wise GMM</text><text class="s" x="315" y="179">shared-variance prior</text>
    <rect class="bx o" x="450" y="75" width="190" height="60" rx="10"/><text x="545" y="100">Clean score</text><text class="s" x="545" y="117">0.7 kNN + 0.3 GMM, EMA</text>
    <rect class="bx o" x="690" y="75" width="190" height="60" rx="10"/><text x="785" y="100">Relabel + class floor</text><text class="s" x="785" y="117">top 20% per class kept</text>
    <rect class="bx g" x="690" y="205" width="190" height="52" rx="10"/><text x="785" y="228">Weights, labels, E_c(t)</text><text class="s" x="785" y="244">clean mass per class</text>
    <rect class="bx" x="450" y="205" width="190" height="52" rx="10"/><text x="545" y="228">M-step: train</text><text class="s" x="545" y="244">weighted GCE loss</text>
    <path class="ar" d="M180 46H228"/><path class="ar" d="M180 166H228"/>
    <path class="ar" d="M400 46L448 92"/><path class="ar" d="M400 166L448 118"/>
    <path class="ar" d="M640 105H688"/><path class="ar" d="M785 135V203"/><path class="ar" d="M690 231H642"/>
    <path class="ar" d="M450 231H95V194"/><text class="s" x="270" y="250">updated model, next epoch</text>
  </svg>
  </div>
</section>

<section id="method">
  <h2>How it handles noise and imbalance</h2>
  <div class="grid g2">
    <div class="card"><span class="tag t-ben">NOISE</span><h4>Two independent witnesses</h4>
      <p><b>kNN agreement</b> in a frozen feature space checks whether a sample's neighbours share its label. This does not depend on a classifier already biased by noise.<br><br><b>Class-wise GMM</b> splits each class's training loss into clean and noisy components. A shared-variance prior stabilises classes with few samples, and a separation check stops it from labelling half of a genuinely clean class as noisy.</p></div>
    <div class="card"><span class="tag t-ben">NOISE</span><h4>Relabelling and a robust loss</h4>
      <p>A label is changed only when the smoothed model and the frozen-feature neighbours both confidently agree on a different class. Training then uses generalised cross-entropy, which is bounded, so a wrong label cannot dominate the gradient.</p></div>
    <div class="card"><span class="tag t-pre">IMBALANCE</span><h4>Per-class floor</h4>
      <p>The top 20% most trustworthy samples in every class always keep a weight of at least 0.5. Rare classes cannot be wiped out by the noise filter.</p></div>
    <div class="card"><span class="tag t-pre">IMBALANCE</span><h4>Clean-effective logit adjustment</h4>
      <p>The effective clean mass per class, E_c(t), is recomputed each epoch and used as the class prior for logit adjustment. Rebalancing follows verified clean signal, not noisy counts, so it does not amplify wrong labels.</p></div>
  </div>
  <h3>Training setup</h3>
  <p class="lead">ResNet-18 (ImageNet init), Adam at 1e-4, batch 128, up to 40 epochs with early stopping on clean validation macro-F1. Metric: macro-F1, which weights rare classes equally. A cross-entropy baseline with static class weights is the comparison.</p>
</section>

<section id="examples">
  <h2>Examples at 30% label noise</h2>
  <p class="lead">Images whose training label was flipped to a wrong class. In each case the baseline was wrong and the proposed model recovered the true diagnosis. Click Run live to score the image with both loaded models.</p>

  <h3>Measured on the 5,319 flipped training samples (30% uniform global noise)</h3>
  <div class="card"><table>
    <tr><th>Metric</th><th>Baseline</th><th>Proposed</th></tr>
    <tr><td>Predicts the TRUE label on flipped samples</td><td>62.1%</td><td class="right">68.7%</td></tr>
    <tr><td>Accuracy on samples with clean labels</td><td>66.9%</td><td class="right">85.9%</td></tr>
    <tr><td>Predicts the (wrong) given label on flipped samples</td><td>7.2%</td><td>11.4%</td></tr>
  </table></div>
  <p class="note">854 flipped samples were wrong under the baseline and right under the proposed model; 31 of them are rare classes (DF, VASC, SCC).</p>

  <h3>Imbalance: a rare class recovered (squamous cell carcinoma)</h3>
  <div class="grid g3" id="ex-rare"></div>
  <h3>Noise: a common class recovered (melanocytic nevus)</h3>
  <div class="grid g3" id="ex-common"></div>
  <p class="note" style="margin-top:12px">These images come from the training split, so they show label-noise recovery, not generalisation. Rare-class recoveries are fewer because under global noise few rare-class images get flipped in the first place.</p>
</section>

<section id="demo">
  <h2>Live demo</h2>
  <p class="lead">Upload a dermoscopy image to see the diagnosis from the proposed model next to the baseline.</p>
  <div id="drop"><b>Click to upload</b> or drag and drop an image<p class="note">JPG or PNG</p>
    <img id="preview" alt="preview"><input id="file" type="file" accept="image/*" hidden></div>
  <div id="status"></div>
  <div id="results"></div>
</section>

<footer>Research demo only. Not a medical device and not a substitute for professional diagnosis.</footer>

<script>
const CODES = ['MEL','NV','BCC','AK','BKL','DF','VASC','SCC'];
const DIST = [['NV',12875],['MEL',4522],['BCC',3323],['BKL',2624],['AK',867],['SCC',628],['VASC',253],['DF',239]];
const DZ = [
 ['MEL','Melanoma','mal','Malignant','Cancer of pigment cells. The most dangerous common skin cancer; it can spread, so early detection is critical.'],
 ['BCC','Basal cell carcinoma','mal','Malignant','The most common skin cancer. Grows slowly and rarely spreads, but can damage surrounding tissue.'],
 ['SCC','Squamous cell carcinoma','mal','Malignant','Cancer of keratinocytes, often on sun-exposed skin. Can spread if untreated. Rare in this dataset.'],
 ['AK','Actinic keratosis','pre','Pre-cancerous','Rough, scaly patch from long-term sun damage. Some progress to squamous cell carcinoma.'],
 ['NV','Melanocytic nevus','ben','Benign','The common mole. The majority class, and the main visual look-alike for melanoma.'],
 ['BKL','Benign keratosis','ben','Benign','Includes seborrheic keratosis and solar lentigo. Harmless but can mimic melanoma.'],
 ['DF','Dermatofibroma','ben','Benign','Firm, harmless skin nodule, often on the legs. The rarest class here.'],
 ['VASC','Vascular lesion','ben','Benign','Blood-vessel lesions such as angiomas. Rare in this dataset.']];
const RARE = [
 ['ISIC_0028659','SCC','BCC','BCC','33%','76%'],
 ['ISIC_0058451','SCC','BKL','BCC','27%','70%'],
 ['ISIC_0030844','SCC','MEL','BKL','27%','68%']];
const COMMON = [
 ['ISIC_0031156','NV','BKL','DF','26%','99%'],
 ['ISIC_0031054','NV','AK','BCC','19%','99%'],
 ['ISIC_0055695','NV','DF','MEL','22%','98%']];

document.getElementById('dist').innerHTML = DIST.map(d =>
  `<div class="r"><span>${d[0]}</span><div class="b"><i style="width:${(d[1]/12875*100).toFixed(1)}%"></i></div><span>${d[1].toLocaleString()}</span></div>`).join('');

document.getElementById('dz').innerHTML = DZ.map(d =>
  `<div class="card"><span class="tag t-${d[2]}">${d[3]}</span><h4>${d[1]} (${d[0]})</h4><p>${d[4]}</p></div>`).join('');

function exCard(e) {
  return `<div class="card ex"><img src="/sample/${e[0]}" alt="${e[0]}" loading="lazy">
    <div class="row"><span>Image</span><span>${e[0]}</span></div>
    <div class="row"><span>True diagnosis</span><span class="right">${e[1]}</span></div>
    <div class="row"><span>Given (wrong) label</span><span class="wrong">${e[2]}</span></div>
    <div class="row"><span>Baseline said</span><span class="wrong">${e[3]} (${e[4]})</span></div>
    <div class="row"><span>Proposed said</span><span class="right">${e[1]} (${e[5]})</span></div>
    <button class="btn small" style="margin-top:8px" onclick="runSample('${e[0]}')">Run live</button></div>`;
}
document.getElementById('ex-rare').innerHTML = RARE.map(exCard).join('');
document.getElementById('ex-common').innerHTML = COMMON.map(exCard).join('');

const drop = document.getElementById('drop'), file = document.getElementById('file');
const preview = document.getElementById('preview'), statusEl = document.getElementById('status');
const results = document.getElementById('results');
drop.onclick = () => file.click();
file.onchange = () => file.files[0] && go(file.files[0]);
drop.ondragover = e => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = e => { e.preventDefault(); drop.classList.remove('over'); e.dataTransfer.files[0] && go(e.dataTransfer.files[0]); };

function card(title, rows, best) {
  const t = rows[0];
  return `<div class="card res ${best ? 'best' : ''}"><h5>${title}</h5>
    <div class="top">${t.name}</div>
    <div class="conf">${t.code} &middot; ${(t.prob*100).toFixed(1)}% confidence</div>
    ${rows.map((r,i) => `<div class="pr ${i===0?'first':''}"><span>${r.name}</span>
      <div class="b"><i style="width:${(r.prob*100).toFixed(1)}%"></i></div><span>${(r.prob*100).toFixed(1)}%</span></div>`).join('')}</div>`;
}

async function predict(fd, previewUrl) {
  preview.src = previewUrl; preview.style.display = 'inline-block';
  results.innerHTML = ''; statusEl.textContent = 'Analysing...';
  try {
    const r = await fetch('/predict', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) { statusEl.textContent = d.error || 'Something went wrong'; return; }
    statusEl.textContent = '';
    results.innerHTML = card('Proposed model', d.proposed, true) + card('Baseline model', d.baseline, false);
  } catch (e) { statusEl.textContent = 'Could not reach the server'; }
}
function go(f) { const fd = new FormData(); fd.append('image', f); predict(fd, URL.createObjectURL(f)); }
function runSample(n) {
  const fd = new FormData(); fd.append('sample', n);
  document.getElementById('demo').scrollIntoView();
  predict(fd, '/sample/' + n);
}
</script>
</body></html>"""


@app.route('/')
def index():
    return PAGE


if __name__ == '__main__':
    print(f"Open http://localhost:{args.port}")
    app.run(host='0.0.0.0', port=args.port)