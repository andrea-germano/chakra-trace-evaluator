import os
import json
import yaml
import itertools
import shutil

# --- 1. BASE CONFIGURATIONS ---

base_mlsynth = {
    "model": {
        "name": "scenario_id_placeholder", # Will be overwritten
        "num_layers": 8,
        "hidden_size": 4096,
        "vocab_size": 32000,
        "bytes_per_val": 2
    },
    "prefill_parallelism": {"tp_size": 1, "pp_size": 1},
    "decode_parallelism": {"tp_size": 1, "pp_size": 1},
    "inference": {
        "serialize_decode_iterations": True,
        "kv_transfer": {
            "mode": "bulk",       
            "direction": "push",
            "explicit_request": True
        },
        "requests": [
            {"prompt_len": 4096, "gen_len": 8} 
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
    "bandwidth": [25], 
    "latency": [500]
}

# --- 2. SEARCH SPACE DEFINITION (SWEEP) ---

sweep_params = {
    "prompt_len": [1024, 4096, 16384],
    "bandwidth": [25, 50, 100],
    "kv_mode": ["bulk", "streaming"]
}

# --- 3. UTILITY FUNCTIONS ---

def clean_and_create_dir(dir_path):
    """Deletes the directory (and its contents) if it exists, then recreates it empty."""
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def generate_configs():
    # Define target directories
    mlsynth_dir = "configs/mlsynth/auto_generated"
    astrasim_dir = "configs/astra_sim/auto_generated"

    print("=> Cleaning up old configurations...")
    clean_and_create_dir(mlsynth_dir)
    clean_and_create_dir(astrasim_dir)

    # Get all possible combinations
    keys = sweep_params.keys()
    values = sweep_params.values()
    combinations = list(itertools.product(*values))

    print(f"=> Generating {len(combinations)} new scenarios...\n")

    for combo in combinations:
        params = dict(zip(keys, combo))
        
        p_len = params["prompt_len"]
        bw = params["bandwidth"]
        mode = params["kv_mode"]

        scenario_id = f"bw{bw}_prompt{p_len}_{mode}"

        # --- UPDATE MLSYNTH ---
        mlsynth_config = json.loads(json.dumps(base_mlsynth)) 
        mlsynth_config["model"]["name"] = scenario_id  # Use scenario_id as the model name
        mlsynth_config["inference"]["requests"][0]["prompt_len"] = p_len
        mlsynth_config["inference"]["kv_transfer"]["mode"] = mode

        # --- UPDATE ASTRA-SIM NETWORK ---
        network_config = json.loads(json.dumps(base_network))
        network_config["bandwidth"] = [bw]
        
        # --- UPDATE ASTRA-SIM SYSTEM ---
        system_config = json.loads(json.dumps(base_system))

        # --- FILE SAVING ---
        mlsynth_path = f"{mlsynth_dir}/{scenario_id}_mlsynth.yaml"
        system_path = f"{astrasim_dir}/{scenario_id}_system.json"
        network_path = f"{astrasim_dir}/{scenario_id}_network.yml"

        # Write YAML (MLSynth)
        with open(mlsynth_path, 'w') as f:
            yaml.dump(mlsynth_config, f, default_flow_style=False, sort_keys=False)

        # Write YAML (Network)
        with open(network_path, 'w') as f:
            yaml.dump(network_config, f, default_flow_style=False, sort_keys=False)

        # Write JSON (System)
        with open(system_path, 'w') as f:
            json.dump(system_config, f, indent=2)

        print(f"Created: {scenario_id}")

if __name__ == "__main__":
    generate_configs()
    print("\nGeneration completed successfully.")