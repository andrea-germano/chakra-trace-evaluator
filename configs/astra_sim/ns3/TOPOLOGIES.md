# Simulator Topologies

This document describes the physical topologies used for the inference workload simulations in ns-3 / ASTRA-sim.

T1, T2 and T2.1 consist of **8 NPUs (GPUs)** configured to execute mixed parallelism strategies (e.g., Tensor Parallelism `TP=2` combined with Pipeline/Data Parallelism). T3 and T4 scale the same design to **12** and **20 NPUs** to model disaggregated inference, where the prefill pool and the decode pool run at *different* TP degrees.

---

## T1 — Hierarchical Multi-Server Architecture (PCIe + 200G Uplink)

This topology models a realistic cluster of 4 servers (or compute "islands"), where each server hosts 2 GPUs. It is a three-tier hierarchical (Multi-Rail) topology:

1. **Intra-Pair (NVLink):** Inside each server, the two GPUs are connected to each other via a direct, ultra-high-bandwidth link (**4800 Gbps**, 0.0005ms). This link is dimensioned to fully absorb the heavy All-Reduce traffic required for Tensor Parallelism (assuming TP=2).
2. **Intra-Server (PCIe Bus):** Both GPUs on the same server communicate with a local PCIe switch (nodes 8, 9, 10, 11 in the DAG) via a fast **1024 Gbps** (0.001ms) connection, modeling a PCIe Gen5 bus, for example.
3. **Inter-Server (Uplink/NIC):** The PCIe switch of each server connects to the central Top-of-Rack (ToR) switch (node 12) via a single network cable limited to **200 Gbps** (0.005ms). This link represents the external bottleneck (e.g., Infiniband or RoCEv2) over which inter-server traffic will travel (KV Cache transfer, Pipeline Parallelism, etc.).

| Server (Island) | GPU (NPU) | Local PCIe Switch | Uplink vs Central Switch |
|:---------------:|:---------:|:-----------------:|:------------------------:|
| A               | 0, 1      | Switch 8          | 200 Gbps                 |
| B               | 2, 3      | Switch 9          | 200 Gbps                 |
| C               | 4, 5      | Switch 10         | 200 Gbps                 |
| D               | 6, 7      | Switch 11         | 200 Gbps                 |

---

## T2 — Simplified "Flat" Network (Single 100G Switch)

This topology represents a more basic network infrastructure where the intermediate tier of local server PCIe switches is removed. The goal is to test scenarios with a less performant "core" network.

1. **Intra-Pair (NVLink):** Just like in T1, the high-performance direct bridge between GPU pairs (0-1, 2-3, etc.) is maintained at **4800 Gbps** (0.0005ms) to offload Tensor Parallelism.
2. **Inter-Server (Flat Network):** There are no local server switches. All 8 GPUs are connected directly to a single central switch (Switch 8) with direct **100 Gbps** (0.005ms) links.

While in T1 two GPUs shared the same 200 Gbps uplink to the outside by pre-aggregating their traffic on the PCIe switch, in T2 every single GPU has its own dedicated 100 Gbps link to the central switch. This creates a more uniform network environment, but one that is individually slower (a 100G bottleneck per NPU on global collective calls).

| GPU Pair (NPU) | NVLink Bridge (Bandwidth) | Connection to Central Switch |
|:--------------:|:-------------------------:|:----------------------------:|
| 0, 1           | 4800 Gbps                 | Direct, 100 Gbps per GPU     |
| 2, 3           | 4800 Gbps                 | Direct, 100 Gbps per GPU     |
| 4, 5           | 4800 Gbps                 | Direct, 100 Gbps per GPU     |
| 6, 7           | 4800 Gbps                 | Direct, 100 Gbps per GPU     |

---

## T2.1 — NVSwitch Pairs + Leaf/Spine (8 NPUs)

Same 8 GPUs and same GPU pairs as T2, but with two substitutions. The **direct NVLink cable becomes a shared NVSwitch**, and the **flat central switch becomes a two-tier leaf/spine network**. Everything else is held equal, so T2.1 isolates the cost of the extra switching tier — and, more importantly, it unlocks scale-up domains larger than 2 GPUs, which T3 and T4 will exploit.

1. **Scale-Up (NVSwitch):** Each GPU pair attaches to its own NVSwitch (nodes 8-11) at **4800 Gbps** (0.0005ms). A shared switch, not a cable: a domain is no longer capped at 2 GPUs.
2. **Scale-Out (NIC → Leaf):** Every GPU also has a dedicated **100 Gbps** (0.00001ms) NIC link into a private leaf switch (nodes 13-20). Each GPU is therefore dual-homed.
3. **Spine:** All 8 leaves uplink to the central spine switch (node 12) at **100 Gbps** (0.005ms).

Traffic inside a pair stays on the 4800 Gbps plane (2 hops); anything leaving a pair crosses the 100 Gbps plane (4 hops). All four domains have the same size, so a single uniform `tp_size = 2` works machine-wide.

| GPU Pair (NPU) | NVSwitch | Domain size | Max TP | Leaf switches | Spine |
|:--------------:|:--------:|:-----------:|:------:|:-------------:|:-----:|
| 0, 1           | 8        | 2           | 2      | 13, 14        | 12    |
| 2, 3           | 9        | 2           | 2      | 15, 16        | 12    |
| 4, 5           | 10       | 2           | 2      | 17, 18        | 12    |
| 6, 7           | 11       | 2           | 2      | 19, 20        | 12    |

---

## T3 — Disaggregated Inference, Prefill TP=4 (12 NPUs)

The same leaf/spine blueprint as T2.1, but the NVSwitches are **unequally populated**. This is the first disaggregated topology: a **prefill pool of 8 GPUs** in two wide TP=4 domains, and a **decode pool of 4 GPUs** in two narrow TP=2 domains.

1. **Scale-Up (NVSwitch):** 4 NVSwitches (nodes 12-15) at **4800 Gbps** (0.0005ms) — two carrying **4 GPUs** (prefill), two carrying **2 GPUs** (decode).
2. **Scale-Out (NIC → Leaf):** 12 private leaf switches (nodes 17-28), **100 Gbps** (0.00001ms) per GPU.
3. **Spine:** node 16, 12 uplinks at **100 Gbps** (0.005ms).

The consequence of the asymmetry is that **no single `tp_size` is valid for the whole machine**: prefill runs at `tp=4`, decode at `tp=2`. The KV cache must therefore be **resharded 4 → 2** as it crosses the spine.

| Pool    | GPU (NPU)  | NVSwitch | Domain size | Max TP | Role               |
|:-------:|:----------:|:--------:|:-----------:|:------:|:------------------:|
| Prefill | 0, 1, 2, 3 | 12       | 4           | 4      | PP stage 0, `tp=4` |
| Prefill | 4, 5, 6, 7 | 13       | 4           | 4      | PP stage 1, `tp=4` |
| Decode  | 8, 9       | 14       | 2           | 2      | PP stage 0, `tp=2` |
| Decode  | 10, 11     | 15       | 2           | 2      | PP stage 1, `tp=2` |

---

## T4 — Disaggregated Inference, Prefill TP=8 (20 NPUs)

Structurally identical to T3, with the prefill domains **widened from 4 to 8 GPUs**. The prefill pool grows to 16 GPUs while the decode pool stays at 4, pushing the asymmetry to its extreme.

1. **Scale-Up (NVSwitch):** 4 NVSwitches (nodes 20-23) at **4800 Gbps** (0.0005ms) — two carrying **8 GPUs** (prefill), two carrying **2 GPUs** (decode).
2. **Scale-Out (NIC → Leaf):** 20 private leaf switches (nodes 25-44), **100 Gbps** (0.00001ms) per GPU.
3. **Spine:** node 24, 20 uplinks at **100 Gbps** (0.005ms).

This is the heaviest **KV resharding** case (**8 → 2**) and the one where the scale-out plane is most strained: 16 GPUs of prefill compute must push their KV cache to the decode pool through a 100 Gbps link each.

| Pool    | GPU (NPU) | NVSwitch | Domain size | Max TP | Role               |
|:-------:|:---------:|:--------:|:-----------:|:------:|:------------------:|
| Prefill | 0 – 7     | 20       | 8           | 8      | PP stage 0, `tp=8` |
| Prefill | 8 – 15    | 21       | 8           | 8      | PP stage 1, `tp=8` |
| Decode  | 16, 17    | 22       | 2           | 2      | PP stage 0, `tp=2` |
| Decode  | 18, 19    | 23       | 2           | 2      | PP stage 1, `tp=2` |

---

## Summary

| | **T1** | **T2** | **T2.1** | **T3** | **T4** |
|:--|:--:|:--:|:--:|:--:|:--:|
| **NPUs** | 8 | 8 | 8 | 12 | 20 |
| **Scale-up (4800 Gbps)** | NVLink cable | NVLink cable | NVSwitch | NVSwitch | NVSwitch |
| **Domain sizes** | 2, 2, 2, 2 | 2, 2, 2, 2 | 2, 2, 2, 2 | **4, 4, 2, 2** | **8, 8, 2, 2** |
| **Max TP** | 2 | 2 | 2 | **4 / 2** | **8 / 2** |
| **Scale-out** | PCIe 1024G → ToR, 200 Gbps shared by 2 GPUs | Flat switch, 100 Gbps per GPU | Leaf/spine, 100 Gbps per GPU | Leaf/spine, 100 Gbps per GPU | Leaf/spine, 100 Gbps per GPU |
| **Hops out of domain** | 3 | 2 | 4 | 4 | 4 |
| **KV resharding** | — | — | — | **4 → 2** | **8 → 2** |