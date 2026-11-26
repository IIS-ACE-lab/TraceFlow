# backend/status_store.py
from utils.timestamps import now_iso

pipeline_status = {
    "status": "Idle (awaiting trigger)",
    "stage": "none",
    "timestamp": now_iso()
}
