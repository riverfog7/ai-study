#!/usr/bin/env python3

import runpod
import os
import dotenv
import json
import shutil
import time

GPU_ID = 'NVIDIA GeForce RTX 5090'
# GPU_ID = 'NVIDIA RTX A5000'
POD_NAME = 'aisogang_dev'
POD_ID_SAVE='./.pod_id'
WAIT_THRESHOLD = 60
POLLING = 1

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
    min_memory_in_gb=20,
    min_download=1000,
    min_vcpu_count=6,
    support_public_ip=True,
    allowed_cuda_versions=["12.8"],
)

pod_id = pod['id']
with open(POD_ID_SAVE, 'w') as f:
    f.write(pod_id)
print(f"Pod created with ID: {pod_id}")
print()
start_time = time.time()
while True:
    pod_status = runpod.get_pod(pod_id)
    if pod_status.get("runtime") and pod_status.get("runtime").get("ports"):
        print("Pod is ready with public IP and ports.")
        break
    else:
        if time.time() - start_time > WAIT_THRESHOLD:
            print("Pod creation is taking too long. Terminating...")
            runpod.terminate_pod(pod_id)
            os.remove(POD_ID_SAVE)
            exit(1)
        time.sleep(POLLING)
print('\n'.join(f"{val['privatePort']} -> http://{val['ip']}:{val['publicPort']}" for val in pod_status['runtime']['ports'] if val['type'] == 'tcp' and val['isIpPublic']))

jupyter_port = 8888
external_jupyter_port = -1
for val in pod_status['runtime']['ports']:
    if val['privatePort'] == jupyter_port and val['type'] == 'tcp' and val['isIpPublic']:
        external_jupyter_port = val['publicPort']
        break
if external_jupyter_port == -1:
    print(f"Failed to find public port mapping for Jupyter port {jupyter_port}.")
    runpod.terminate_pod(pod_id)
    os.remove(POD_ID_SAVE)
    exit(1)
jupyter_url = f"http://{pod_status['runtime']['ports'][0]['ip']}:{external_jupyter_port}/lab"
os.system(f"printf {jupyter_url} | pbcopy")
