import os
import json
import yaml
import itertools

# --- 1. CONFIGURAZIONI BASE ---

base_mlsynth = {
    "model": {
        "name": "llama-test",
        "num_layers": 8,
        "hidden_size": 4096,
        "vocab_size": 32000,
        "bytes_per_val": 2
        # RIMOSSO: sequence_len
    },
    "prefill_parallelism": {"tp_size": 1, "pp_size": 1},
    "decode_parallelism": {"tp_size": 1, "pp_size": 1},
    "inference": {
        "serialize_decode_iterations": True,
        "kv_transfer": {
            "mode": "bulk",       # Verrà sovrascritto dal loop
            "direction": "push",
            "explicit_request": True
        },
        "requests": [
            {"prompt_len": 4096, "gen_len": 8} # Verrà sovrascritto dal loop
        ]
    }
}

base_system = {
    "scheduling-policy": "LIFO",
    "peak-perf": 989,
    "local-mem-bw": 3350,
    "all-reduce": "ring",
    "reduce-scatter": "ring",
    "all-gather": "ring",
    "roofline-enabled": 1
}

base_network = {
    "topology": ["Switch"],
    "npus_count": [2],
    "bandwidth": [25], # Verrà sovrascritto dal loop
    "latency": [500]
}

# --- 2. DEFINIZIONE DELLO SPAZIO DI RICERCA (SWEEP) ---

sweep_params = {
    "prompt_len": [1024, 4096, 16384],
    "bandwidth": [25, 50, 100],
    "kv_mode": ["bulk", "streaming"]
}

# --- 3. FUNZIONI UTILITY ---

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_configs():
    # Ottieni tutte le combinazioni possibili
    keys = sweep_params.keys()
    values = sweep_params.values()
    combinations = list(itertools.product(*values))

    print(f"Generazione di {len(combinations)} scenari in corso...\n")

    for combo in combinations:
        # Crea un dizionario per la configurazione corrente
        params = dict(zip(keys, combo))
        
        # Estrai i parametri per comodità
        p_len = params["prompt_len"]
        bw = params["bandwidth"]
        mode = params["kv_mode"]

        # Genera un ID scenario coerente
        scenario_id = f"bw{bw}_prompt{p_len}_{mode}"

        # ---------------------------------------------------
        # UPDATE MLSYNTH
        # ---------------------------------------------------
        mlsynth_config = json.loads(json.dumps(base_mlsynth)) # Deep copy veloce
        mlsynth_config["inference"]["requests"][0]["prompt_len"] = p_len
        mlsynth_config["inference"]["kv_transfer"]["mode"] = mode

        # ---------------------------------------------------
        # UPDATE ASTRA-SIM NETWORK
        # ---------------------------------------------------
        network_config = json.loads(json.dumps(base_network))
        network_config["bandwidth"] = [bw]
        
        # ---------------------------------------------------
        # UPDATE ASTRA-SIM SYSTEM (resta invariato per ora)
        # ---------------------------------------------------
        system_config = json.loads(json.dumps(base_system))

        # ---------------------------------------------------
        # SALVATAGGIO FILE
        # ---------------------------------------------------
        mlsynth_path = f"configs/mlsynth/auto_generated/{scenario_id}_mlsynth.yaml"
        system_path = f"configs/astra_sim/auto_generated/{scenario_id}_system.json"
        network_path = f"configs/astra_sim/auto_generated/{scenario_id}_network.yml"

        ensure_dir(mlsynth_path)
        ensure_dir(system_path)
        ensure_dir(network_path)

        # Scrivi YAML (MLSynth)
        with open(mlsynth_path, 'w') as f:
            yaml.dump(mlsynth_config, f, default_flow_style=False, sort_keys=False)

        # Scrivi YAML (Network)
        with open(network_path, 'w') as f:
            yaml.dump(network_config, f, default_flow_style=False, sort_keys=False)

        # Scrivi JSON (System)
        with open(system_path, 'w') as f:
            json.dump(system_config, f, indent=2)

        print(f"Creato: {scenario_id}")

if __name__ == "__main__":
    generate_configs()
    print("\nGenerazione completata con successo.")