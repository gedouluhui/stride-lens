# StrideLens

**Your tensors can share storage without sharing a single element. See the difference.**

StrideLens reads real PyTorch CPU tensors and produces a self-contained, interactive layout report. Trace an index to its storage-relative byte offset, see broadcast aliases, compare slices, and inspect actual `view()` attempts. No model, GPU, API key, server, or remote assets.

```python
import torch
from stridelens import capture, write_report

x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
report = capture({"even": x[:, ::2], "odd": x[:, 1::2]})
print(report["pairs"][0])
# same_storage_region=True, overlap='no', overlap_bytes=0
write_report(report, "layout.html")
```

Open `layout.html` locally. Select a tensor, click a logical element, or enter its index. All addresses are relative to anonymous storage regions; tensor values and raw process addresses are not exported.

## Try it

From a checkout, in your own Python environment:

```sh
python -m pip install .
# Capture/demo requires PyTorch. If you do not already have it:
python -m pip install '.[capture]'
python -m stridelens demo --out demo.html --json demo.json
```

The demo uses seven synthetic tensors, including disjoint slices, a real reshape copy, a zero-stride broadcast, and an empty view. You can also download [the generated demo HTML](docs/demo.html) and open it locally without installing anything.

Rendering saved evidence does **not** import PyTorch:

```sh
python -m stridelens render demo.json --out another-report.html
```

Existing outputs are protected; add `--force` only when you intend to replace them. HTML and JSON writes are individually atomic, not a multi-file transaction.

## Investigate a view

```python
snapshot = capture(
    {"transpose": x.t(), "strided slice": x[:, ::2]},
    view_requests={"transpose": [-1], "strided slice": [-1]},
)
# Flattening the transpose is rejected by PyTorch.
# The noncontiguous strided slice can still be viewed as [6].
write_report(snapshot, "views.html")
```

View attempts call real `tensor.view()` and never automatically make a contiguous copy. Adjacent merge barriers explain stride discontinuities; they are not a complete causal diagnosis for every rejected view. Named tensors, for example, may reject a view for another reason.

## What the evidence means

| Report field | Meaning |
|---|---|
| Same storage region | Same nonempty base pointer and byte extent during one live capture; not Python object identity |
| Overlap: yes/no | Exact addressed-byte intersection, or a proof from disjoint bounds/empty tensors |
| Overlap: unknown | Enumeration was capped; this is **not** a no-overlap result |
| Internal overlap | Multiple logical indices address the same bytes; exact only when enumerated |
| Offsets | Bytes from the tensor's storage start, not absolute memory addresses |

Different external storage wrappers can still overlap, including unaligned buffers. Pair analysis accounts for bytes, rather than merely comparing first-element pointers or storage IDs.

## Deliberate limits

- Accepts exact `torch.Tensor`, CPU, strided float32. No CUDA, sparse, quantized, meta, conjugate/negative flags, or tensor subclasses. Explicitly use `parameter.detach()` if appropriate; StrideLens does not do it silently.
- At most 32 tensors, 16 axes, 128-character names. Enumeration defaults to at most 4,096 elements per tensor; `max_elements` can lower that limit. Large tensors retain metadata and conservative pair conclusions.
- The page displays 64 logical cells at a time. Enter any valid index above the grid to select it and jump to its page; Previous/Next browse contiguous index ranges. Use Left/Right and Home/End from the grid's single keyboard entry. Display paging never changes the analysis budget.
- No tensor values, model/source code, or raw addresses are exported automatically. **Names and shapes remain metadata you may consider sensitive.** Choose labels carefully before sharing.
- JSON validation checks types, bounds, view-address sequence equivalence even beyond the enumeration budget, and consistency of pair conclusions involving provably identical address sets. The view check uses a bounded symbolic floor-sum representation, not element sampling. It cannot authenticate who created a snapshot or independently prove missing cross-region addresses. Reports are not signed attestations.
- This is not a profiler: no GPU allocation timeline, kernel-copy detection, latency measurement, or speedup claim.

Python 3.10+ is the supported syntax floor. Local behavior was exercised on Python 3.12.3 / torch 2.12.0+cpu. Other versions and real-user usefulness require independent verification; a workflow file alone is not a successful CI run.

## Why not another tensor visualizer?

[Interactive tensor-layout explanations](https://guillesanbri.com/tensor-layouts/) are excellent for learning. [PyTorch memory tools](https://pytorch.org/blog/understanding-gpu-memory-1/) address allocation history. StrideLens focuses on a small, shareable report from **your actual tensors**, with bounded byte-overlap conclusions and explicit uncertainty. If printing `shape` and `stride()` already answers your question, use that simpler solution.

## Development and verification

```sh
python -m pip install build setuptools wheel
python scripts/verify.py
python -m build --no-isolation
python scripts/check_sdist.py dist/stridelens-0.1.1.tar.gz
```

Optional real-browser regression (fresh isolated contexts, no shared profile):

```sh
python -m pip install '.[browser]'
python -m playwright install chromium
python scripts/check_browser.py
# Or use an already installed Edge, without downloading a browser:
python scripts/check_browser.py --channel msedge
```

The browser check exercises 360/1280px viewports, arbitrary-index paging, keyboard navigation, visible/accessible labels, reduced-motion controls, unsafe labels, and a real JSON download saved inside a temporary project directory. Missing browser dependencies fail explicitly, rather than count as passing tests.

The tests use real CPU tensors, direct scalar addresses as an independent mapping reference, known disjoint/overlapping cases, unaligned shared buffers, malformed reports, and CLI subprocesses. Review regressions include empty huge-axis allocation and contradictory metadata. Tests are not a proof of correctness or a measure of adoption.

To contribute: open an Issue with a minimal **synthetic** tensor example, torch/Python version, expected outcome, and actual result. Never include private weights or raw memory dumps. Add a failing regression test before changing behavior. Be respectful and focus discussion on reproducible evidence; maintainers may remove abusive or unrelated content.

## 中文简介

把真实 CPU 张量的形状、步长和共享关系变成可离线打开的交互报告。区分“同一存储”与“实际重叠”，用真实 PyTorch 验证 view 请求；默认不导出张量值或原始内存地址。大张量超出枚举预算时明确显示未知，不宣称测得显存或性能优势。

## License and provenance

MIT. Original implementation developed with AI assistance and reviewed through explicit regression tests. PyTorch is an optional separately installed dependency, not bundled. The linked explanatory articles and upstream projects are references, not code/assets copied into this repository.
