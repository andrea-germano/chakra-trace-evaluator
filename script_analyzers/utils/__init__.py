"""
utils — shared machinery for the MLSynth sweep analyzers.

    paths    every path, derived from one --sweep name; and the swept-axis parsing.
    roles    the declared rank -> role map, and flow classification from it.
    fabric   what the ns-3 switch does: topology, config, the PFC/ECN physics.
             Reads no simulation output; predicts from design parameters alone.
    ns3      readers for fct.txt / pfc.txt / qlen.txt.
    flows    fct rows + topology + placement -> classified, path-annotated flows.
    astra    readers for the ASTRA-sim stats_sys*.csv.
    plots    the plotting mechanics (not the plots).

The analyzers on top answer different questions and stay separate:

    bandwidth_sweep.py   how does the run scale with link bandwidth?
                         Lives entirely in the ASTRA CSVs: more bandwidth ->
                         shorter transfers, monotone, visible in the ticks alone.
    buffer_sweep.py      does the switch buffer change when decode can start?
                         An ns-3 question: at steady state the link drains at line
                         rate whatever the buffer is, so the CSVs say nothing
                         happens. What moves is the congestion REGIME.
    ns3_analyzer.py      one run, in the time domain. The sweeps collapse each run
                         to scalars; every question that makes a sweep hard to
                         read ("does the queue peak WHILE PFC pauses?") is a
                         question about when, and no max() survives it.

Nothing here interprets a run. Interpretation is what the analyzers do
differently, and it is deliberately not shared.

    python3 -m utils.fabric <topology> <config> --bottleneck 8->12 --buffers 2,4,8
    python3 -m utils.roles --from-astra <astra_run_dir>
"""