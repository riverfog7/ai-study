#!/usr/bin/env python3
import runpod
import os
import dotenv

POD_ID_SAVE='./.pod_id'

dotenv.load_dotenv(dotenv.find_dotenv())
if not runpod.check_credentials():
    runpod.set_credentials(os.getenv("RUNPOD_API_KEY"))
    runpod.api_key = os.getenv("RUNPOD_API_KEY")

if os.path.exists(POD_ID_SAVE):
    with open(POD_ID_SAVE, 'r') as f:
        pod_id = f.read().strip()
    try:
        runpod.terminate_pod(pod_id)
    except Exception as e:
        print(f"Failed to terminate pod {pod_id}: {e}")
    os.remove(POD_ID_SAVE)