import os
import json
import time
from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import config

app = FastAPI()

# Files for inter-process communication
STATE_FILE = os.path.join(os.path.dirname(__file__), "feroke_state.json")
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "feroke_settings.json")

# Ensure settings file exists with defaults
if not os.path.exists(SETTINGS_FILE):
    default_settings = {
        "min_green": config.MIN_GREEN_TIME,
        "max_green": config.MAX_GREEN_TIME,
        "max_clearance": config.MAX_CLEARANCE_TIME,
        "zones": [config.ZONE_A, config.ZONE_B, config.ZONE_C, config.ZONE_D]
    }
    with open(SETTINGS_FILE, "w") as f:
        json.dump(default_settings, f)

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/status")
async def get_status():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {"error": "Inference engine not running"}

@app.get("/api/settings")
async def get_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {}

@app.post("/api/settings")
async def update_settings(settings: dict = Body(...)):
    # Read existing
    current = {}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                current = json.load(f)
        except json.JSONDecodeError:
            pass
    
    # Update
    current.update(settings)
    
    # Save (Inference engine watches this file)
    temp_file = SETTINGS_FILE + ".tmp"
    with open(temp_file, "w") as f:
        json.dump(current, f)
    os.replace(temp_file, SETTINGS_FILE)
    
    return {"status": "success", "settings": current}

@app.get("/api/zones")
async def get_zones():
    settings = await get_settings()
    return settings.get("zones", [])

@app.post("/api/zones")
async def set_zones(zones: list = Body(...)):
    settings = await get_settings()
    # Ensure zones are lists of lists for JSON
    cleaned_zones = []
    for zone in zones:
        cleaned_zone = [[int(p[0]), int(p[1])] for p in zone]
        cleaned_zones.append(cleaned_zone)
    
    settings["zones"] = cleaned_zones
    temp_file = SETTINGS_FILE + ".tmp"
    with open(temp_file, "w") as f:
        json.dump(settings, f)
    os.replace(temp_file, SETTINGS_FILE)
    
    return {"status": "success", "zones": cleaned_zones}

if __name__ == "__main__":
    print(f"Admin UI running at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
