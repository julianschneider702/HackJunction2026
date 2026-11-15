# station/ - scripts for the station laptop (Ubuntu, Intel Core Ultra)

| Script | What it does |
|---|---|
| `studio-python` | Runs a script with the Python of Studio's backend: `station/studio-python sorter/policy_sort.py` |
| `start-studio.sh` | Starts Physical AI Studio (backend + UI), then open http://localhost:3000 |
| `install-intel-gpu.sh` | Installs the Intel GPU drivers PyTorch XPU needs: `sudo bash station/install-intel-gpu.sh` |
| `xpu-usage.sh` | Live load of the Arc iGPU per engine and process, without sudo (`--once` measures once) |
| `fix-cameras.sh` | After re-plugging cameras: writes their new USB fingerprints into Studio so they show online again |

Studio is expected next to this repo (`../physical-ai-studio`). Set `PHYSICAL_AI_STUDIO` to use another checkout.
