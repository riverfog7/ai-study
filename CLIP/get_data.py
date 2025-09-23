#%%
import json
import os.path
import random
import time
import urllib.parse

from playwright.sync_api import sync_playwright

req_delay = 0.5
req_jitter = 0.2
targets = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a bird",
    "a plane in the sky",
    "a car",
    "a truck",
    "a ship",
    "a bottle of soda",
    "a cup of coffee",
    "a bowl of soup",
    "a plate of food",
    "a keyboard",
    "a mouse",
    "a monitor",
    "a laptop",
    "a book",
    "a chair",
    "a table",
    "a bed",
    "a person",
    "a tree",
    "a flower",
    "a mountain",
    "a river",
    "a beach",
    "a cityscape",
    "a sunset",
    "a sunrise",
    "a rainbow",
    "a starry night",
    "a snowy landscape",
    "a rainy day",
    "a foggy morning",
]
req_template = "https://www.google.com/search?tbm=isch&q={}"

images = {val: [] for val in targets}
if os.path.exists("clip_image_urls.json"):
    with open("clip_image_urls.json", "r") as f:
        images = json.load(f)
with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    ctx = browser.new_context(
        viewport={ "width": 3840, "height": 2160 },
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Safari/605.1.15")
    page = ctx.new_page()
    for target in targets:
        if target in images and len(images[target]) > 0:
            continue
        url = req_template.format(target)
        url = urllib.parse.quote(url, safe=":/?=&")
        page.goto(url)
        # handle consent dialogs if present, then scroll
        for _ in range(5):
            page.mouse.wheel(0, 2000)
            page.wait_for_timeout(1000)
        imgs = page.locator("img").all()
        images[target] = [img.get_attribute("src") for img in imgs if img.get_attribute("src")]
        print(f"Found {len(images[target])} images for '{target}'")
        time.sleep(req_delay + random.uniform(-req_jitter, req_jitter))

        with open("clip_image_urls.json", "w") as f:
            json.dump(images, f, indent=2)
    browser.close()

for target in images:
    images[target] = [val for val in images[target] if "encrypted-tbn" in val and "images" in val]
    if not os.path.exists(f'./sample_images/{target.replace(" ", "_")}/'):
        os.makedirs(f'./sample_images/{target}/')
    with open(f'./sample_images/{target}/urls.txt', 'w') as f:
        for url in images[target]:
            f.write(url + '\n')

with open("clip_image_refined.json", "w") as f:
    json.dump(images, f, indent=2)
