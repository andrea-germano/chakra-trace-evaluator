"""
utils — shared machinery for the MLSynth sweep analyzers.

    fabric   what the ns-3 switch does: topology, config, the PFC/ECN physics.
             Reads no simulation output; predicts from design parameters alone.
    ns3      readers for fct.txt / pfc.txt / qlen.txt.
    astra    readers for the ASTRA-sim stats_sys*.csv.
    sweep    run discovery and the swept-axis parsing.
    plots    the plotting mechanics (not the plots).

The analyzers on top answer different questions and stay separate:

    bandwidth_analyzer.py   how does the run scale with link bandwidth?
                            Lives entirely in the ASTRA CSVs.
    buffer_analyzer.py      does the switch buffer change when decode can start?
                            Crosses two levels: predicts the congestion regime from
                            fabric, then checks it against the ns-3 outputs.

Nothing here interprets a run. Interpretation is what the two analyzers do
differently, and it is deliberately not shared.

    python3 -m utils.fabric <topology> [config] --buffers 2,4,8,16,32
"""