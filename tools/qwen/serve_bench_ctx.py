"""usage: serve_bench.py PORT MODEL_ID LOGFILE DEPTH[,DEPTH...] [--vision]. Real text prompts of ~DEPTH tokens (unique offset per depth, no prefix-cache reuse)."""
import sys, json, time, base64, io, urllib.request, os, re
port, mid, logf, depths = sys.argv[1:5]
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "exllamav3", "conversion", "standard_cal_data")
def post(body, t=3600):
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps({"model": mid, **body}).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=t))
def last_req():
    for l in reversed(open(logf).read().splitlines()):
        if "qserve: request" in l: return l.split("qserve: ")[1]
    return ""
txt = "".join(open(f"{D}/{f}", encoding="utf8").read() for f in ("wiki.utf8", "c4.utf8", "technical.utf8", "code.utf8"))
for i, d in enumerate(int(x) for x in depths.split(",")):
    chars = int(d * 4.2); off = 12345 * (i + 1) + d
    prompt = (txt * 3)[off:off + chars]
    t0 = time.time()
    r = post({"messages": [{"role": "user", "content": prompt + "\n\nWrite a long, detailed continuation of the story about a lighthouse keeper (at least 300 words)."}],
              "max_tokens": 256, "temperature": 0, "enable_thinking": False})
    out = r["choices"][0]["message"]["content"]; u = r["usage"]
    words = out.split(); loop = len(words) > 40 and len(set(zip(words, words[1:], words[2:]))) < 0.5 * (len(words) - 2)
    print(f"RESULT depth~{d} prompt={u['prompt_tokens']} new={u['completion_tokens']} wall={time.time()-t0:.1f}s loop={loop} | {last_req()}", flush=True)
if "--vision" in sys.argv:
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (448, 448), (255, 255, 255)); d = ImageDraw.Draw(im)
    d.ellipse((60, 60, 220, 220), fill=(220, 30, 30)); d.rectangle((250, 240, 400, 380), fill=(30, 60, 220)); d.text((120, 400), "HELLO 42", fill=(0, 0, 0))
    b = io.BytesIO(); im.save(b, "PNG"); url = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
    r = post({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "Describe the shapes, their colors and any text in this image."}]}], "max_tokens": 200, "temperature": 0, "enable_thinking": False})
    print("VISION:", r["choices"][0]["message"]["content"][:300].replace("\n", " "), "|", last_req(), flush=True)
