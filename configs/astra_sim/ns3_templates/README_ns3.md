# Backend ns-3 — configurazione (trace_evaluator)

Questi file configurano il backend **ns-3** (packet-level) di ASTRA-sim, usato al posto
dell'analitico per simulare la rete sulle tracce Chakra generate da MLSynth.
Tutti i percorsi nei file sono **assoluti** (`/home/andre/tesi/trace_evaluator/...`),
quindi il binario può essere lanciato da qualsiasi directory.

```
configs/astra_sim/ns3_templates/
    simple_topology.txt     # topologia FISICA  (NPU, switch, link)
    logical_1D.json         # topologia LOGICA  (come girano le collettive)
    ns3_config.txt          # config del SIMULATORE (CC, ECN, banda, output)
    flow.txt                # flussi manuali  -> 0
    trace.txt               # nodi da monitorare -> 0
output/ns3/                 # qui finiscono fct.txt, pfc.txt, qlen.txt, mix.tr
```

**Vincolo centrale**: tutto ruota intorno al numero di NPU = numero di file `.et`
generati da MLSynth = `tp_p·pp_p + tp_d·pp_d`. Con il config `test` (prefill 1×2,
decode 2×2) sono **6 NPU**.

---

## simple_topology.txt — topologia fisica

```
7 1 6
6
6 0 200Gbps 0.005ms 0
6 1 200Gbps 0.005ms 0
...
6 5 200Gbps 0.005ms 0
```

Formato (parsato con `>>`, **niente commenti né righe extra**):

- **Riga 1** — `<nodi_totali> <num_switch> <num_link>`
  Qui `7 1 6`: 6 NPU (ID 0–5) + 1 switch (ID 6) = 7 nodi, 1 switch, 6 link.
- **Riga 2** — gli ID dei nodi che sono switch. Qui `6`.
- **Righe successive** — un link per riga: `<src> <dst> <banda> <ritardo> <error_rate>`.

I nodi compute (le NPU) hanno gli ID bassi `0 .. N-1`; gli switch gli ID successivi.
Il numero di nodi compute (`nodi_totali − num_switch`) **deve** essere uguale al numero di NPU.

Parametri per link che puoi cambiare:

- **banda** (`200Gbps`) → capacità del link. Deve avere una entry nelle mappe ECN del
  config (vedi sotto), altrimenti ns-3 crasha su un assert.
- **ritardo** (`0.005ms`) → latenza del link; incide sui tempi di trasferimento KV cache.
- **error_rate** (`0`) → lascialo a 0 salvo studi specifici sulla perdita di pacchetti.

---

## logical_1D.json — topologia logica

```json
{ "logical-dims": ["6"] }
```

Il **prodotto** delle dimensioni deve fare il numero totale di NPU.
Disaccoppia *come* le collettive vengono mappate sulle NPU dalla topologia fisica:
è la ragione per cui ns-3 distingue logico e fisico (nell'analitico coincidevano).

- `["6"]` → 1-D: tutte le 6 NPU in un'unica dimensione.
- `["3","2"]` → 2-D: prima collettiva su gruppi da 3, poi stride da 2.

Vincolo: il numero di dimensioni logiche deve eguagliare la lunghezza delle liste
`*-implementation` nel system config (1-D ↔ liste da 1 elemento, 2-D ↔ da 2, ecc.).

---

## ns3_config.txt — config del simulatore

Le manopole che probabilmente toccherai:

- `CC_MODE 12` → algoritmo di congestion control. **Nota**: i valori documentati sono
  `1`=DCQCN, `3`=HPCC, `7`=TIMELY, `8`=DCTCP, `10`=HPCC-PINT; il `12` è il default
  degli esempi di questo fork ns-3 — se vuoi un algoritmo noto, verifica la gestione
  di `cc_mode` nel sorgente del backend e passa a uno dei valori documentati.
- `PACKET_PAYLOAD_SIZE 1000` → dimensione pacchetto in byte.
- `ENABLE_TRACE 0` → run leggero. Mettilo a `1` per il dump pacchetto-livello in
  `mix.tr` (e popola `trace.txt` con i nodi da monitorare).
- `KMAX_MAP / KMIN_MAP / PMAX_MAP` → soglie ECN **per banda di link**. Devi avere una
  entry per ogni banda usata nella topologia (qui sono coperti 25/40/100/200/400/2400 Gbps).
- `*_OUTPUT_FILE`, `QLEN_MON_FILE` → dove finiscono i log pacchetto-livello (`output/ns3/`).

Il resto (timer DCQCN, finestra, ecc.) lascialo ai default finché non studi quegli effetti.

---

## flow.txt / trace.txt

File di **input**, mai sovrascritti.

- `flow.txt = 0` → nessun flusso manuale; ASTRA-sim inietta i propri dal workload.
- `trace.txt = 0` → nessun nodo monitorato. Per monitorarne alcuni: `2` su una riga,
  poi `0 1` (gli ID), con `ENABLE_TRACE 1`.

---

## Rappresentare diverse topologie di rete

La topologia fisica vive tutta in `simple_topology.txt`. Cambiando le tre righe di
intestazione e l'elenco dei link puoi modellare strutture diverse. Esempi:

### Single-switch (quella attuale)

Tutte le NPU su un unico switch — semplice e simmetrica, ideale per il sanity check.
Per **più NPU** basta aumentare i nodi:

```
9 1 8            # 8 NPU + 1 switch
8
8 0 200Gbps 0.005ms 0
... fino a 8 7 ...
```

Aggiorna `nodi_totali`, `num_link`, l'elenco link, e `logical-dims` → `["8"]`.

### Fat-tree / Clos (due livelli)

Due tier di switch: **leaf/ToR** (collegati alle NPU) e **spine** (collegano i ToR tra loro).
Serve a studiare la congestione cross-node, rilevante quando prefill e decode stanno su
nodi diversi. Esempio minimale con 8 NPU, 2 ToR (4 NPU ciascuno), 2 spine, mesh completa:

```
12 4 12          # 8 NPU + 4 switch = 12 nodi ; 12 link
8 9 10 11        # 8,9 = ToR ; 10,11 = spine
0 8 200Gbps 0.005ms 0     # NPU 0-3 -> ToR 8
1 8 200Gbps 0.005ms 0
2 8 200Gbps 0.005ms 0
3 8 200Gbps 0.005ms 0
4 9 200Gbps 0.005ms 0     # NPU 4-7 -> ToR 9
5 9 200Gbps 0.005ms 0
6 9 200Gbps 0.005ms 0
7 9 200Gbps 0.005ms 0
8 10 200Gbps 0.0125ms 0   # ToR -> spine (link "superiori")
8 11 200Gbps 0.0125ms 0
9 10 200Gbps 0.0125ms 0
9 11 200Gbps 0.0125ms 0
```

I link ToR→spine hanno tipicamente un ritardo maggiore. Con un fat-tree di solito si usa
una `logical-dims` 2-D (es. `["4","2"]`: prima dimensione intra-ToR, seconda inter-ToR),
e il system config va portato a 2-D. Esempi reali più grandi sono i file
`128_nodes_16_switch_topology.txt` e `128_nodes_32_switch_topology.txt` di ASTRA-sim.

### Ring (senza switch)

NPU collegate direttamente in anello, `num_switch = 0` (la riga 2 resta vuota):

```
8 0 8

0 1 25Gbps 0.005ms 0
1 2 25Gbps 0.005ms 0
...
7 0 25Gbps 0.005ms 0
```

Utile per isolare il costo delle collettive in topologie a banda ridotta senza switch.

---

## Checklist quando cambi config MLSynth o topologia

1. Conta le NPU: `tp_p·pp_p + tp_d·pp_d` (lo script lo stampa dopo MLSynth).
2. Aggiorna `simple_topology.txt`: riga 1, elenco switch, righe di link.
3. Aggiorna `logical-dims` perché il prodotto faccia il numero di NPU.
4. Porta le liste `*-implementation` del system config alla stessa dimensionalità.
5. Verifica che ogni banda di link sia presente nelle mappe `KMAX/KMIN/PMAX`.