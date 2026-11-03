"""usage: gate_texts.py PORT MODEL_ID OUT.json DEPTH[,DEPTH...] [N=128]: three greedy answers (temperature 0, N tokens) per depth to real-text
prompts of ~DEPTH tokens, saved for a text-equality comparison of two server builds/flags (gate_texts.py A.json B.json compare mode:
gate_texts.py --cmp A.json B.json)."""
import sys, json, time, os, urllib.request
if sys.argv[1] == "--cmp":
    A, B = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
    for k in A:
        for i, (a, b) in enumerate(zip(A[k], B[k])):
            div = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), None)
            print(f"depth {k} prompt {i}: {'IDENTICAL' if div is None and len(a) == len(b) else 'DIVERGES at char ' + str(div)} ({len(a)}/{len(b)} chars)")
    sys.exit(0)
port, mid, out, depths = sys.argv[1:5]; n = int(sys.argv[5]) if len(sys.argv) > 5 else 128
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "exllamav3", "conversion", "standard_cal_data")
def post(body, t=3600):
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps({"model": mid, **body}).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=t))
srcs = [open(f"{D}/{f}", encoding="utf8").read() for f in ("wiki.utf8", "c4.utf8", "technical.utf8")]
res = {}
for d in (int(x) for x in depths.split(",")):
    res[str(d)] = []
    for i, s in enumerate(srcs):
        off = 4000 * (i + 1) + 77 * d % 5000
        prompt = s[off:off + int(d * 4.2)]
        r = post({"messages": [{"role": "user", "content": prompt + "\n\nSummarize the text above in three sentences."}], "max_tokens": n, "temperature": 0, "enable_thinking": False})
        res[str(d)].append(r["choices"][0]["message"]["content"])
        print(f"depth {d} prompt {i}: {r['usage']['prompt_tokens']} prompt tokens, {len(res[str(d)][-1])} chars", flush=True)
json.dump(res, open(out, "w"))
