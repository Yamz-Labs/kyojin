"""Labelled corpus for the answer-phase loop guard (tools/qwen/loopguard_gate.py, test_guard.py).

legit(): long valid repetitive outputs (code, tables, JSON, HTML, real model answers kept in fixtures/loopguard/legit).
loops(): degenerate answers, real ones from past runs (fixtures/loopguard/loops) continued the way a loop continues,
plus synthetic ones. Nothing here needs a GPU.

The model answers in fixtures/loopguard/legit come from the public benchmark report of issue #109 (CC-BY-4.0, Yamz-Labs/kyojin
user report): one-shot pagoda / hexagon pages written by the two engines. loops/ holds answers of earlier release-gate runs.
"""
import glob
import json
import os
import random
import re
from pathlib import Path
HERE = Path(__file__).resolve().parent
FX = str(HERE / 'fixtures' / 'loopguard') + '/'
def legit():
    out={}
    for f in sorted(glob.glob(FX+'legit/*')):
        out[os.path.basename(f)]=open(f).read()
    rng=random.Random(7)
    out['json_records']=json.dumps([{"id":i,"name":f"user{i}","email":f"user{i}@example.com","active":i%3==0,"tags":["a","b","c"],"score":round(rng.random(),4)} for i in range(400)],indent=2)
    out['json_compact']=json.dumps([{"id":i,"type":"point","x":rng.randint(0,99),"y":rng.randint(0,99)} for i in range(600)])
    out['md_table']="| id | name | qty | price |\n|---|---|---|---|\n"+"\n".join(f"| {i} | item-{i} | {rng.randint(1,9)} | {rng.randint(10,99)}.{rng.randint(0,99):02d} |" for i in range(300))
    out['csv']="id,name,value\n"+"\n".join(f"{i},row{i},{rng.randint(0,9999)}" for i in range(600))
    out['css']="\n".join(f".c{i} {{ margin: {i%8}px {i%5}px; padding: 4px 8px; color: #{rng.randrange(1<<24):06x}; background: #{rng.randrange(1<<24):06x}; border-radius: 4px; }}" for i in range(150))
    out['css_utility']="\n".join(f".m-{i} {{ margin: {i}px; }}\n.p-{i} {{ padding: {i}px; }}\n.w-{i} {{ width: {i}%; }}" for i in range(80))
    out['html_list']="<ul>\n"+"\n".join(f'  <li class="item"><a href="/page/{i}">Page {i}</a></li>' for i in range(300))+"\n</ul>"
    out['html_table']="<table>\n"+"\n".join(f'  <tr><td class="cell">{i}</td><td class="cell">name{i}</td><td class="cell">{rng.randint(0,99)}</td></tr>' for i in range(250))+"\n</table>"
    out['js_array']="const verts = [\n"+",\n".join(f"  [{rng.uniform(-1,1):.3f}, {rng.uniform(-1,1):.3f}, {rng.uniform(-1,1):.3f}]" for i in range(500))+"\n];"
    out['js_calls']="\n".join(f"addBox({rng.randint(0,30)}, {rng.randint(0,30)}, {rng.randint(0,30)}, {rng.randint(1,5)}, {rng.randint(1,5)}, 0x{rng.randrange(1<<24):06x});" for i in range(300))
    out['js_setters']="\n".join(f"el{i}.style.left = '{i*10}px'; el{i}.style.top = '{i*7}px'; el{i}.style.width = '40px'; el{i}.style.height = '40px';" for i in range(200))
    out['py_asserts']="\n".join(f"    assert parse('{i}') == {i}, 'parse failed for {i}'\n    assert fmt({i}) == '{i}', 'fmt failed for {i}'" for i in range(150))
    out['py_defs']="\n\n".join(f"def get_field_{i}(self):\n    return self._data['field_{i}']\n\ndef set_field_{i}(self, value):\n    self._data['field_{i}'] = value" for i in range(60))
    out['svg_path']="".join(f'<circle cx="{rng.randint(0,500)}" cy="{rng.randint(0,500)}" r="{rng.randint(2,9)}" fill="#{rng.randrange(1<<24):06x}"/>\n' for i in range(300))
    out['c_switch']="switch (op) {\n"+"\n".join(f"    case OP_{i}: result = handle_{i}(a, b); break;" for i in range(200))+"\n}"
    out['voxel_dense']="\n".join("function fill%s(cx,cz,r,y0,y1,color){for(let y=y0;y<=y1;y++)for(let x=cx-r;x<=cx+r;x++)for(let z=cz-r;z<=cz+r;z++){const dx=x-cx,dz=z-cz;if(dx*dx+dz*dz<=r*r+%d.4)setVoxel(x,y,z,%s);}}"%(n,k,c) for k,(n,c) in enumerate(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",["color","pick(cols)"]*13)))
    out['numbered_steps']="\n".join(f"{i}. Step {i}: take the value from register {i} and add it to the accumulator, then store the result." for i in range(1,61))
    out['config_yaml']="\n".join(f"  - name: svc-{i}\n    image: registry/app:{i}\n    replicas: 2\n    port: {8000+i}" for i in range(100))
    out['sql_inserts']="\n".join(f"INSERT INTO t (id, name, v) VALUES ({i}, 'name{i}', {rng.randint(0,999)});" for i in range(300))
    out['hex_dump']="\n".join(f"{i*16:08x}  "+" ".join(f"{rng.randrange(256):02x}" for _ in range(16)) for i in range(250))
    # repetitive data that is valid: a zero grid, identical records, a hex dump of zeroed memory (the period rule used to cut them)
    out['zero_grid']="GRID = [\n"+",\n".join("    ["+", ".join(["0"]*40)+"]" for _ in range(80))+"\n]\n"
    out['zero_grid_json']="[\n"+",\n".join("  ["+",".join(["0"]*80)+"]" for _ in range(80))+"\n]\n"
    out['json_lines_identical']="\n".join('{"status": "ok", "code": 200, "message": "request completed", "retry": false}' for _ in range(60))+"\n"
    out['hex_dump_zero']="\n".join(f"{i*16:08x}  "+" ".join(["00"]*16)+"  |................|" for i in range(250))+"\n"
    out['hex_dump_zero_plain']="\n".join(f"{i*16:08x}: "+"0000 "*8 for i in range(250))+"\n"
    # repo sources as long legit code
    for f in ['qwen/serve.py', 'glm/serve.py', 'qwen/test_guard.py']:    # long valid code written by people
        out['src_' + f.replace('/', '_')] = (HERE.parent / f).read_text()[:30000]
    return out
def synth_loops():
    out={}
    out['code_loop_exact']="function init(){\n  setup();\n}\n"+"    renderTile(x, y, z); tick++;\n"*400
    out['code_loop_fn']="".join(f"function step(){{ update(); draw(); requestAnimationFrame(step); }}\n" for _ in range(200))
    out['drift_counter']="\n".join(f"Then the scene adds another lantern at position {i} and keeps the same lighting and shadows." for i in range(300))
    out['drift_words']="".join(f'The pagoda has a roof tier "{i}" with curved eaves and red pillars supporting the level above it. ' for i in range(150))
    out['json_loop']='{"items":['+",".join('{"id":%d,"type":"lamp","color":"red","pos":[1,2,3]}'%(i%3) for i in range(300))+"]}"
    out['sentence_loop']="I think the answer is yes. But wait, let me reconsider. "*120
    out['para_loop']="\n\n".join(["The model should output the final answer now. Let me finalize the response and summarize the key points once more."]*60)
    out['list_loop']="\n".join(f"- Consider option {1+i%4}: check the constraint and verify the result again." for i in range(300))
    out['html_loop']='<div class="tower"><div class="tier"></div></div>\n'*300
    out['counter_calls']="function build(){\n"+"\n".join(f"  fillBox(cx-1,cx+1,{200+i},{200+i},cz-1,cz+1,[C.stone,C.stoneD,C.stoneL]);" for i in range(400))
    out['char_loop']="ab"*1500
    # the same data kinds as the legit zero grid / JSON records, but endless: they stop at the data backstop (loopguard.DATA_SUSTAIN)
    out['zero_run_endless']="GRID = [\n"+"0, "*40000
    out['json_lines_endless']="\n".join('{"status": "ok", "code": 200, "message": "request completed", "retry": false}' for _ in range(1500))+"\n"
    return out



def extend(t, n=16000):
    """Continue a looping text: its last 500 characters come back again and again, a counter moving each time (no exact
    period, the case the text rule exists for)."""
    seg = t[-500:]
    out = []
    for i in range(n // len(seg) + 1):
        out.append(re.sub(r"\b[a-z]+\b", lambda m, c=[0]: (c.__setitem__(0, c[0] + 1), f"{m.group(0)}{i}" if c[0] == 7 else m.group(0))[1], seg, count=8))
    return t + "".join(out)


def extend_shuffled(t, n=16000, seed=0):
    """Continue a looping text by re-emitting the sentences / lines of its last 1200 characters in random order, a counter
    moving now and then: the loop never repeats a stretch longer than a unit (the drift the text rule exists for)."""
    rng = random.Random(seed)
    units = [u for u in re.split(r"(?<=[.!?\n])\s*", t[-1200:]) if len(u) > 3] or [t[-1200:]]
    out, tot, k = [], 0, 0
    while tot < n:
        u = rng.choice(units)
        if rng.random() < 0.3:
            u = re.sub(r"\d+", str(k), u, count=1)
        out.append(u)
        tot += len(u) + 1
        k += 1
    return t + " " + " ".join(out)


def loops():
    files = sorted(glob.glob(FX + "loops/*"))
    real = {os.path.basename(f) + "+cont": extend(open(f).read()) for f in files}
    real.update({os.path.basename(f) + "+shuf": extend_shuffled(open(f).read(), seed=i) for i, f in enumerate(files)})
    real.update({os.path.basename(f): open(f).read() for f in files if os.path.basename(f).startswith('pagoda_')})   # as generated
    return {**synth_loops(), **real}
