#!/usr/bin/env python3

import runpod
import os
import dotenv
import json
import shutil

GPU_ID = 'NVIDIA RTX A6000'
POD_NAME = 'aisogang_dev'
POD_ID_SAVE='./.pod_id'

shutil.copyfile('./configure.sh', '/Volumes/riverfog7.com/web/files/ai-study-configure.sh')
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

pod = runpod.create_pod(
    gpu_type_id=GPU_ID,
    image_name='',
    template_id='dbd0hf7hma',
    gpu_count=1,
    name=f"{POD_NAME}",
    cloud_type='COMMUNITY',
    volume_mount_path='/workspace',
    container_disk_in_gb=80,
    volume_in_gb=0,
    min_memory_in_gb=50,
    min_download=1000,
    min_vcpu_count=8,
)

pod_id = pod['id']
with open(POD_ID_SAVE, 'w') as f:
    f.write(pod_id)
print(f"Pod created with ID: {pod_id}")
