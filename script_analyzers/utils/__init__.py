"""
utils — shared machinery for the MLSynth sweep analyzers.

    paths    every path, derived from one --sweep name; the swept-axis parsing;
             and the cross-model workload discovery the companions share.
    cli      the Abort/need convention every analyzer raises through, and the
             single warn/drain_warnings stream they all speak into.
    roles    the declared rank -> role map, and flow classification from it.
    fabric   what the ns-3 switch does: topology, config, the PFC/ECN physics.
             Reads no simulation output; predicts from design parameters alone.
    ns3      readers for fct.txt / pfc.txt / qlen.txt.
    flows    fct rows + topology + placement -> classified, path-annotated flows.
    pp       pipeline-parallel activation handoffs: how skewed is the arrival of
             one stage's output at the next, measured from fct.txt + placement.
    astra    readers for the ASTRA-sim stats_sys*.csv, the counted-once transfer
             de-dup, and the shared derived instants (first-token send, roles).
    intervals  interval-set algebra (union / overlap / subtract / concurrency),
             pure geometry over (start, end) pairs; used by astra and the analyzers.
    measures the per-run measures more than one analyzer reports: TTFT, the KV
             barrier and skews (including the per-(stage, layer) shard
             population), the decode-side stall/all-reduce stats and their
             worst-stage reduction, and the per-link score (LinkStat /
             link_metrics / pause_stats). Plus knee_scalars, the one SWEEP-wide
             reading -- "where does the buffer stop mattering" -- parametrised
             by the columns each sweep declares. Measurement only; what a number
             MEANS stays in the analyzers.
    plots    the plotting mechanics (not the plots): series, log-2 axes, save,
             the max-per-bucket queue-series downsampler, the shared palette and
             swept-axis colour ramp, zoom_y, the knee rules, the lossy-run
             overlay and the one-line-per-group figure both cross-model
             companions draw.

The analyzers on top answer different questions and stay separate:

    bandwidth_sweep.py   how does one run scale with link bandwidth?
                         Lives entirely in the ASTRA CSVs: more bandwidth ->
                         shorter transfers, monotone, visible in the ticks alone.
    buffer_sweep.py      does the switch buffer change when decode can start?
                         An ns-3 question: at steady state the link drains at line
                         rate whatever the buffer is, so the CSVs say nothing
                         happens. What moves is the congestion REGIME.
    incast_sweep.py      the same fabric with the KV fan-in as the second axis:
                         buffer_sweep's readings applied per topology (its knees,
                         waterfall, distributions, bloat and per-link spread),
                         plus the two the buffer sweep cannot ask -- how the
                         handover scales with the incast DEGREE, and what the KV
                         flow-completion tail looks like.
    ns3_analyzer.py      one run, in the time domain. The sweeps collapse each run
                         to scalars; every question that makes a sweep hard to
                         read ("does the queue peak WHILE PFC pauses?") is a
                         question about when, and no max() survives it.
    bandwidth_compare.py \\  the cross-MODEL companions: run the matching sweep
    buffer_compare.py    /  analyzer over every workload that ran it and overlay
                         them, one figure per metric. Same readers, same scoring;
                         the only new question is how models differ, not runs.

Nothing here interprets a run. Interpretation is what the analyzers do
differently, and it is deliberately not shared.

    python3 -m utils.fabric <topology> <config> --bottleneck 8->12 --buffers 2,4,8
    python3 -m utils.roles --from-astra <astra_run_dir>
"""