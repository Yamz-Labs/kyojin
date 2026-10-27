# ablit3 prompt sources

- AdvBench harmful behaviors (`goal` column): llm-attacks/llm-attacks data,
  <https://github.com/llm-attacks/llm-attacks> (repo license MIT).
  URL: <https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv>
- HarmBench standard text behaviors (`Behavior` column):
  centerforaisafety/HarmBench,
  <https://github.com/centerforaisafety/HarmBench> (repo license MIT).
  URL: <https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv>
- JailbreakBench JBB-Behaviors harmful split (`Goal` column):
  HF `JailbreakBench/JBB-Behaviors` config `behaviors` split `harmful`
  (project license MIT), fetched via `datasets`.
- XSTest prompts, safe rows only (`prompt` where `label` == safe):
  paul-rottger/xstest, <https://github.com/paul-rottger/xstest>
  (LICENSE = CC-BY-4.0).
  URL: <https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv>
- Alpaca instructions (`instruction` + `input`):
  HF `tatsu-lab/alpaca` (dataset card license CC BY-NC 4.0), fetched via `datasets`.
