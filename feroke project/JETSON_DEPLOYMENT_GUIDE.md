# 🛰️ Jetson Nano Deployment Guide

This guide explains how to deploy and run the **Feroke Vanguard** system on your Jetson Nano using the models trained on the DGX A100 Supercomputer.

## 📁 1. Preparation
You only need to transfer the folder named **`feroke project`** to your Jetson.

### Included Models (In `Feroke-Traffic-Lite/models/`):
- `nano_edge.pt`: The high-speed model (Default).
- `heavy_hpc.pt`: High-accuracy model for testing.

---



## ⚡ 2. Power Settings (Manual Step)
To get the full potential of your Jetson's GPU, you must set it to **Max Power Mode**. Since you are doing this manually, ensure you run these commands on the Jetson terminal first:

```bash
# Set to MAX Power (Architecture dependent, check your manual)
sudo nvpmodel -m 0
# Run jetson_clocks to lock speeds
sudo jetson_clocks
```

---

## 🚀 3. How to Run
Once the folder is on your Jetson:
1. Open the terminal and navigate to the `Feroke-Traffic-Jetson` folder.
2. Run the master script:
   ```bash
   chmod +x run.sh
   ./run.sh
   ```

### What happens inside `run.sh`:
- **First Run**: It will detect that you don't have a `.engine` file yet. It will automatically start compiling your HPC model using the Jetson's GPU. **This takes ~10 minutes.**
- **Subsequent Runs**: It will skip compilation and launch the dashboard immediately.
- **GUI Launch**: It will attempt to open the **Desktop GUI** (Feroke Vanguard Command Center).
- **Fallback**: If no monitor is detected, it will fall back to starting the **Web Dashboard** on port `8000`.

---

## 📊 4. Verifying GPU Performance
To check if the Jetson is actually using its GPU and to monitor temperature:
1. Open a **second terminal**.
2. Run:
   ```bash
   sudo jtop
   ```
   *(If you don't have jtop, install it with `sudo pip3 install jetson-stats`)*
3. **Target FPS**: You should see the Nano model hitting **30-45 FPS** once the `.engine` conversion is finished.

---

## 🛠️ 5. Switching Models
If you want to test the **Heavyweight** model later:
1. Edit `Feroke-Traffic-Lite/config.py`.
2. Change `MODEL_PATH` to `./models/heavy_hpc.pt`.
3. Change `AI_IMG_SIZE` to 640.
4. Delete the existing `.engine` file in the models folder.
5. Re-run `./run.sh`.

> [!TIP]
> Always stick with the **Nano model** for 24/7 bridge monitoring to keep the Jetson cool and responsive!
