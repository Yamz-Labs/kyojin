"""
Generator.iterate_start_jobs calls pre_start_hook(job) right before a job that fits allocates its pages (the prefix disk restore),
and never for a job that has to wait. Pure python: the real method is loaded by ast and run on a fake generator.
"""
import ast, os, sys, textwrap, types
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(ROOT, "exllamav3", "generator", "generator.py")).read()
ns = {}
for n in ast.walk(ast.parse(src)):
    if isinstance(n, ast.FunctionDef) and n.name == "iterate_start_jobs":
        exec(textwrap.dedent(ast.get_source_segment(src, n)), ns)
fn = ns["iterate_start_jobs"]

events = []

class Job:
    def __init__(self, name, need):
        self.name, self.need, self.sequences, self.skips, self.max_skips = name, need, [object()], 0, 3
        self.serial_number, self.identifier = 0, None
    def current_new_pages_required(self): return self.need
    def activate(self): events.append(("activate", self.name))
    def allocate_pages(self): events.append(("allocate", self.name))

pt = types.SimpleNamespace(num_unreferenced_pages=lambda: 10)
gen = types.SimpleNamespace(active_jobs=[], pending_jobs=[Job("a", 4), Job("big", 50), Job("c", 3)], max_batch_size=4, pagetable=pt,
                            reap_failed_job=lambda *a: None, _prefill_leader=lambda job: None, pre_start_hook=lambda job: events.append(("hook", job.name)))
fn(gen, [])
assert events == [("hook", "a"), ("activate", "a"), ("allocate", "a"), ("hook", "c"), ("activate", "c"), ("allocate", "c")], events
assert [j.name for j in gen.pending_jobs] == ["big"]

events.clear()
gen.pending_jobs, gen.active_jobs = [Job("d", 2)], []
del gen.pre_start_hook                                   # no hook installed: unchanged behaviour
fn(gen, [])
assert events == [("activate", "d"), ("allocate", "d")], events
print("start hook ok")
