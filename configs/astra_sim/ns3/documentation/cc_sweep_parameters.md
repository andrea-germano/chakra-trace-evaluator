# CC Parameter Provenance

Where every congestion-control parameter in `config.template.txt` comes from, how it
compares to the primary sources, and which deviations must be declared when presenting
the CC comparison (`cc_sweep`). Verified on 2026-08-03 against the ns-3 backend sources
in `astra-sim/extern/network_backend/ns-3` and the references listed at the bottom.

## Lineage

```
DCQCN paper (SIGCOMM'15)  ─┐
TIMELY paper (SIGCOMM'15) ─┼─►  HPCC artifact (SIGCOMM'19, alibaba-edu repo)
DCTCP paper (SIGCOMM'10)  ─┘         │  simulation/run.py generates per-CC configs
                                     ▼
                          ASTRA-sim ns-3 backend (fork of HPCC simulator)
                                     │  scratch/config/config.txt = flattened single
                                     │  parameter block, some values changed
                                     ▼
                          config.template.txt (this repo)
                                     │  thesis changes: MTU 3500, ECN maps halved and
                                     │  extended, ACK_HIGH_PRIO 1, STRICT_PRIORITY 1,
                                     ▼  HEADROOM_FACTOR 3, parametric buffer
                          cc_sweep/<topo>_bx100_<cc>_buf<N>/config.txt
                                     ▲  per-CC overrides applied on top of the
                                     │  template: HPCC-artifact values at 100G
                                     │  (*_vwin variants) — see "Applied
                                        configuration" below
```

Two consequences worth stating explicitly:

1. **The template alone is NOT the HPCC-artifact configuration.** The artifact's
   `run.py` generates a *different parameter set per CC algorithm* (see table below);
   the template applies one uniform block. For this reason the `cc_sweep` generator
   rewrites the CC-specific keys per algorithm with the artifact's 100G values
   (recorded in each folder's `manifest.json` under `cc_overrides`). The other sweeps
   (bandwidth/buffer/incast/oversub, all DCQCN) still use the plain template.
2. **Parameters not exposed in `config.txt` are inherited from the ns-3 code defaults**
   (`rdma-hw.cc` attribute defaults). This is how TIMELY gets all of its constants.

## Per-parameter table

Template value vs. HPCC artifact (`run.py` at `--bw 100`) vs. primary source.
"code default" = the value in `rdma-hw.cc` that applies when the key is absent.

| Key (template value) | HPCC artifact @100G | Primary source | Verdict |
|---|---|---|---|
| `CC_MODE` per sweep folder | modes 1/3/7/8 + 12 | — | `none`=12 is window-only (no rate control), ASTRA-sim's own default → clean "ideal fabric" baseline |
| `ALPHA_RESUME_INTERVAL 1` (µs) | 1 (mlx set) | Mellanox-firmware behaviour as modelled by HPCC authors (`rdma-hw.cc` "mlx version") | OK |
| `RATE_DECREASE_INTERVAL 4` (µs) | 4 (mlx set) | idem | OK |
| `RP_TIMER 900` (µs) | 300 (mlx) / 55 (paper set) | DCQCN paper §5.3 recommends 55µs timer + 10MB byte counter (byte counter not modelled here); code default 1500 | **Provenance unclear** (ASTRA-sim upstream choice). Same order of magnitude as NIC defaults; slower rate-recovery than the artifact. Declare it. |
| `EWMA_GAIN 0.00390625` (=1/256) | 1/256 for DCQCN, **1/16 for DCTCP** | DCQCN paper §5.3 (Fig. 12) picks g=1/256; DCTCP paper and Linux impl. use g=1/16 | OK for DCQCN; **wrong for DCTCP** (16× slower α adaptation) |
| `FAST_RECOVERY_TIMES 1` | 1 | DCQCN paper fixes F=5 (Table 2); artifact overrides to 1 | Matches artifact |
| `RATE_AI 50Mb/s` | DCQCN 20, HPCC 40, TIMELY 100 | DCQCN paper: RAI fixed 40Mb/s @40G; TIMELY paper: 10Mb/s @10G (artifact scales both linearly with bw) | Same ballpark for all three; not per-CC scaled. Minor. |
| `RATE_HAI 100Mb/s` | DCQCN 200, TIMELY 500 | QCN hyper-increase stage | Same ballpark. Minor. |
| `MIN_RATE 100Mb/s` | 1000Mb/s | — | Lower floor → throttled flows recover from further down. Minor; declare. |
| `DCTCP_RATE_AI 1000Mb/s` | 615Mb/s (= MTU/RTT for 1KB MTU, 13µs RTT, see run.py comment) | DCTCP adds 1 MTU per RTT | For our fabric (MTU 3500B, base RTT ≈21µs) 1 MTU/RTT ≈ 1333Mb/s → 1000 is the right order. OK. |
| `HAS_WIN 1`, `VAR_WIN 1` | dcqcn/timely: 0 (but `*_vwin` variants: 1); hpcc/dctcp: 1 | HPCC paper §5 evaluates both | **Every CC runs with a per-pair-BDP window cap** ([entry.h:145]). Equal-footing choice; corresponds to the artifact's `dcqcn_vwin`/`timely_vwin` variants. Cite it as such. |
| `GLOBAL_T 0` | artifact uses 1 (global max RTT) | — | Per-pair BDP/RTT is *more* accurate on our heterogeneous fabric (NVLink vs 100G planes). OK. |
| `FAST_REACT 1` | 1 for HPCC only | HPCC paper (react per ACK) | Only affects INT-based CC; harmless for others. OK. |
| `U_TARGET 0.95` | 0.95 | HPCC paper η=95% | OK |
| `MI_THRESH 0` | 0 (artifact default) | HPCC paper maxStage | Matches artifact default. OK. |
| `INT_MULTI 1` | bw/25 = **4** @100G | HPCC artifact scales INT byte/qlen units with line rate | **Deviation**: coarser INT precision at 100G than the artifact. Declare or set 4 for hpcc configs. |
| `KMIN/KMAX/PMAX` @100G: 200KB/800KB/0.2 | 400/1600/0.2 (all CC except DCTCP); DCTCP: **300/300/1.0** (step marking) | DCQCN paper §5.3: RED-like marking, small Pmax; NVIDIA reference config for 100GbE RoCE: min 150KB / max 1500KB | Thesis values = artifact halved → earlier marking, still inside the vendor-recommended envelope. Uniform across CCs, so internally consistent — but **DCTCP canonically wants step-at-K marking** (K=65 pkts @10G in the paper), not a RED ramp. |
| `PACKET_PAYLOAD_SIZE 3500` | 1000 | RoCE MTU 4096 class | Realistic; kmin ≈ 57 pkts, fine. |
| `ACK_HIGH_PRIO 1` | 1 for dcqcn/timely, 0 for hpcc/dctcp/vwin | RoCE deployments prioritise CNP | Uniform 1 = fair, realistic. OK. |
| `BUFFER_SIZE {8,16,32}` (MB/switch) | artifact: `16*bw/50` = **32MB @100G** | Broadcom Trident 3 (100G-era leaf ASIC): 32MB fully-shared buffer | 32 = artifact value and real ASIC class; 8/16 = shallow-buffer stress regimes. OK. |
| `HEADROOM_FACTOR 3` | fixed headroom in artifact | PFC headroom sized ≈ BDP×factor ([common.h:735]) | Custom but principled (per-link BDP-proportional). OK. |

## TIMELY: constants are hard-coded, not configurable

`config.txt` has **no keys** for TIMELY's constants; they come from `rdma-hw.cc`
attribute defaults: α=0.875 (EWMA), β=0.8, TLow=50µs, THigh=500µs, minRTT=20µs.
β, TLow, THigh and the additive increment match the TIMELY paper (§6: "Tlow of 50µs,
Thigh of 500µs, additive increment of 10 Mbps, multiplicative decrement factor (β)
of 0.8" — tuned for 10Gbps with 16–64KB segments).

Two observations for the thesis:
- Our leaf/spine base RTT is ≈20–25µs, so `minRtt=20µs` happens to match the fabric.
- TLow=50µs means TIMELY tolerates ≈25–30µs of queueing (~350KB at 100G) before the
  gradient reacts: expect visibly higher queues than HPCC. This is a **known property
  of paper-tuned TIMELY at 100G**, reported in the HPCC paper as well — a result, not
  a configuration bug.

## Applied configuration (cc_sweep)

The `cc_sweep` configs are generated with the artifact's per-CC values at 100G,
applied as line overrides on top of the template. Citable as: **"congestion-control
parameters as in the HPCC SIGCOMM'19 artifact at 100 Gbps (windowed `*_vwin`
variants), with ECN thresholds per the artifact's linear-scaling formulas; DCTCP's
additive increment recomputed for this fabric's MTU and base RTT."**

Common (all CCs): `RP_TIMER 300`, `MIN_RATE 1000Mb/s` (artifact values, replacing
upstream's 900/100).

| Key | dcqcn | hpcc | hpcc-pint | timely | dctcp | none |
|---|---|---|---|---|---|---|
| `EWMA_GAIN` | 1/256 | 1/256 (unused) | 1/256 (unused) | 1/256 (unused) | **1/16** | — |
| `RATE_AI` | 20Mb/s | 40Mb/s | 40Mb/s | 100Mb/s | 10Mb/s (unused) | — |
| `RATE_HAI` | 200Mb/s | 40Mb/s | 40Mb/s | 500Mb/s | 10Mb/s (unused) | — |
| `DCTCP_RATE_AI` | — | — | — | — | **1300Mb/s** (=MTU/RTT: 3500B/21µs) | — |
| `FAST_REACT` | 0 | 1 | 1 | 0 | 0 | 0 |
| `INT_MULTI` | 1 | **4** (=bw/25) | **4** | 1 | 1 | 1 |
| `PINT_LOG_BASE` / `PINT_PROB` | — | — | **1.05 / 1.0** (ε=0.025, all packets) | — | — | — |
| ECN @100G | 400/1600KB, p=0.2 | same (ignored) | same (ignored) | same (ignored) | **K=300KB step** (kmin=kmax, p=1.0) | same (ignored) |

ECN maps for all speeds follow the artifact formulas (standard: kmin=100·bw/25 KB,
kmax=400·bw/25 KB; DCTCP: kmin=kmax=30·bw/10 KB), with the 1024G/4800G (PCIe/NVLink)
entries kept effectively non-marking. Step marking for DCTCP is safe in this backend:
with kmin==kmax the `q > kmax` branch fires first and the ramp branch is unreachable
(`switch-mmu.cc:100-114`).

## Scale-up plane exemption (NVLink/PCIe outside every CC's domain)

The 4800G (NVLink/NVSwitch) and 1024G (PCIe, T1) links model scale-up planes
that in reality run their own link-level flow control and do not participate in
RoCE congestion control. Each algorithm is kept off that plane by its own
per-hop mechanism:

- **HPCC**: switches do not push INT hop info on ports whose rate the INT
  header cannot encode (backend patch: guard in `switch-node.cc` around
  `PushHop`, helper `IntHop::RateEncodable` in `int-header.h`). Before this
  patch, 4800G ports fell into the encoder's `default:` branch, were read back
  as 25G, and HPCC throttled all TP traffic to the floor — any hpcc run
  produced before the patch is invalid and must be re-run after rebuilding.
- **DCQCN / DCTCP**: the 4800G/1024G entries of the ECN maps are effectively
  non-marking (kmin 7MB). Empirically the guard is never even approached: the
  per-port queue peak on NVSwitches is 0.00 MB across the verified cc_sweep
  runs — the 4800G plane simply does not queue against a 100G fabric.
- **TIMELY**: the RTT signal is end-to-end and cannot be exempted per hop, but
  is de-facto exempt: NVLink base RTT is ~2µs and the observed NVLink queueing
  is zero, far below TLow=50µs, so TIMELY never reacts to the scale-up plane.
- **none**: window-only by construction.
- **hpcc-pint**: same guard as HPCC applied to the `m_ccMode == 10` branch —
  scale-up hops do not contribute to the PINT power estimate, so the max-U the
  sender decodes is over fabric hops only. (PINT never had the 25G-mis-encoding
  bug — it computes with the true port bitrate — but without the guard the
  NVLink hop would still, incorrectly, be sampled.) An NVLink-only flow decodes
  the minimum power (`Pint::decode_u(0) = 1/max_concurrent > 0`, no
  division-by-zero in `UpdateRateHpPint`) and rides at line rate under
  window+PFC, like every other exempted CC.

The discipline is deliberately per-hop, not per-QP: a flow that egresses via a
scale-up NIC can still cross the fabric (T1: PCIe -> 200G ToR), and there the
CC must stay active. Encodable INT rates are 25/50/100/200/400G — every fabric
speed actually used by any sweep; a non-encodable rate on a genuine fabric link
(e.g. a hypothetical bx40 HPCC run) is reported once per rate at simulation
start ("INT: rate ... not encodable"), and one free slot remains in
`lineRateValues[7]` to add it.

## What to declare in the thesis (honest-methods paragraph)

1. Parameters follow the HPCC SIGCOMM'19 artifact at 100G; all CCs run with a
   per-destination window capped at the path BDP (the artifact's `*_vwin` setting),
   and PFC remains enabled — i.e. we model lossless RoCEv2 with each CC on top.
2. Deliberate deviations from the artifact, all documented above: `GLOBAL_T 0`
   (per-pair BDP/RTT — more accurate on a heterogeneous NVLink+Ethernet fabric than
   the artifact's global max-RTT), MTU 3500B (RoCE 4KB class vs the artifact's 1KB),
   `ACK_HIGH_PRIO 1` and `STRICT_PRIORITY 1` uniformly (CNP/ACK prioritisation as in
   RoCE deployments; the artifact toggles ack_prio per CC), `DCTCP_RATE_AI` recomputed
   for this fabric's MTU/RTT, and BDP-proportional PFC headroom (`HEADROOM_FACTOR 3`).
3. TIMELY's constants are hard-coded in the backend at the paper values (§ above);
   its late reaction at 100G (TLow=50µs ≈ 350KB of tolerated queue) is a known
   property of paper-tuned TIMELY, not a mis-configuration.
4. `none` (CC_MODE 12) is the window-only ideal baseline: BDP-capped injection with
   PFC as the only backpressure.

## Sources

- **DCQCN**: Zhu et al., *Congestion Control for Large-Scale RDMA Deployments*,
  SIGCOMM 2015. §5.3: g=1/256, RED-like marking with small Pmax, Kmin=5KB /
  Kmax=200KB @40G, 55µs timers, RAI=40Mb/s.
  <https://conferences.sigcomm.org/sigcomm/2015/pdf/papers/p523.pdf>
- **TIMELY**: Mittal et al., *TIMELY: RTT-based Congestion Control for the
  Datacenter*, SIGCOMM 2015. §6: Tlow 50µs, Thigh 500µs, +10Mb/s, β=0.8.
  <https://conferences.sigcomm.org/sigcomm/2015/pdf/papers/p537.pdf>
- **DCTCP**: Alizadeh et al., *Data Center TCP (DCTCP)*, SIGCOMM 2010. g=1/16,
  K=20 pkts @1G / 65 pkts @10G. <http://ccr.sigcomm.org/online/files/p63_0.pdf>;
  Linux kernel DCTCP doc (shift_g=4 ⇒ g=1/16):
  <https://www.kernel.org/doc/html/v5.9/networking/dctcp.html>
- **HPCC**: Li et al., *HPCC: High Precision Congestion Control*, SIGCOMM 2019.
  η=95%, INT-based, per-ACK reaction. Artifact (per-CC parameter generator, ECN maps,
  buffer=32MB @100G):
  <https://github.com/alibaba-edu/High-Precision-Congestion-Control/blob/master/simulation/run.py>
- **PINT**: Ben Basat et al., *PINT: Probabilistic In-band Network Telemetry*,
  SIGCOMM 2020 — the probabilistic log-encoded telemetry HPCC-PINT replaces INT
  with; log base 1.05 ⇔ ε=0.025 per the HPCC repo's own config documentation.
- **NVIDIA/Mellanox RoCE deployment guide** (ECN thresholds for 100GbE: min 150KB,
  max 1500KB):
  <https://enterprise-support.nvidia.com/s/article/recommended-network-configuration-examples-for-roce-deployment>
- **Broadcom Trident 3** press release (32MB fully-shared packet buffer, 100G-era
  leaf ASIC):
  <https://www.broadcom.com/company/news/product-releases/12056>
- **ASTRA-sim ns-3 backend** (this repo's simulator): code defaults in
  `astra-sim/extern/network_backend/ns-3/src/point-to-point/model/rdma-hw.cc`;
  window assignment in `astra-sim/astra-sim/network_frontend/ns3/entry.h:145`;
  upstream reference config `scratch/config/config.txt`.
