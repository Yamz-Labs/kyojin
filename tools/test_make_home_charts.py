#!/usr/bin/env python3
"""make_home_charts.py: generates every homepage SVG in both themes, valid XML, numbers match the data files, README references them all."""
import glob, json, os, re, subprocess, sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def build(tmp_path):
    subprocess.run([sys.executable, os.path.join(HERE, "make_home_charts.py"), str(tmp_path)], check=True, capture_output=True)
    return sorted(os.listdir(tmp_path))


def test_files_valid_and_paired(tmp_path):
    names = build(tmp_path)
    assert len(names) == 30
    for n in names:
        if n.endswith("-light.svg"): assert n.replace("-light", "-dark") in names
        ET.parse(os.path.join(tmp_path, n))
        assert "Gradient" not in open(os.path.join(tmp_path, n)).read()
    assert open(os.path.join(tmp_path, "decode-mac-light.svg")).read() != open(os.path.join(tmp_path, "decode-mac-dark.svg")).read()


def test_numbers_come_from_data(tmp_path):
    build(tmp_path)
    rd = lambda n: open(os.path.join(tmp_path, n)).read()
    F = json.load(open(os.path.join(ROOT, "doc", "figures.json")))
    q = F["models"][0]
    assert f">{q['decode']['8192']}<" in rd("decode-strix-light.svg") and f"Kyojin {F['version']}" in rd("decode-strix-light.svg")
    assert f">{q['prefill']['8192']:,}<" in rd("prefill-strix-light.svg")
    P = json.load(open(os.path.join(ROOT, "doc", "figures_platforms.json")))
    assert ">74.6<" in rd("decode-rtx-light.svg") and P["rtx"]["qwen27"]["points"][0]["decode"] == 74.6
    assert ">3,001<" in rd("prefill-mac-light.svg")
    assert ">1.7x<" in rd("duel-rtx-qwen27-light.svg")


def test_readme_uses_every_chart(tmp_path):
    names = build(tmp_path)
    readme = open(os.path.join(ROOT, "README.md")).read()
    for n in names:
        assert f"doc/img/home/{n}" in readme, n
    for ref in re.findall(r'doc/img/home/[\w-]+\.svg', readme):
        assert ref.split("/")[-1] in names
