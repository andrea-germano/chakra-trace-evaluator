# Simulator Topologies

This document describes the physical topologies used for the inference workload simulations in ns-3 / ASTRA-sim. 

In all scenarios, the system consists of **8 NPUs (GPUs)** configured to execute mixed parallelism strategies (e.g., Tensor Parallelism `TP=2` combined with Pipeline/Data Parallelism).

---

## T1 — Hierarchical Multi-Server Architecture (PCIe + 200G Uplink)
**File:** `physical_topology_t1.txt`

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
**File:** `physical_topology_t2.txt`

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