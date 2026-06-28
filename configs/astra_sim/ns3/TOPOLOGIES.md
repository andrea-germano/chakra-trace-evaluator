## T1 — Single central switch (reference)

Baseline topology for symmetric serving: prefill `tp2/pp2` and decode `tp2/pp2`,
**8 NPUs**, 4 NVLink islands. Each tensor-parallel pair (2 GPUs) sits on its own
**NVSwitch at 4800 Gbps**; the four NVSwitches hang off a single central fabric
switch. TP all-reduce stays inside the island on NVLink; everything that leaves
an island (KV transfer, PP activations, first-token handoff) crosses the fabric
at `Bx`.

| NPU | pool / stage    | island |
|:---:|:----------------|:------:|
| 0,1 | prefill stage 0 | A      |
| 2,3 | prefill stage 1 | B      |
| 4,5 | decode  stage 0 | C      |
| 6,7 | decode  stage 1 | D      |