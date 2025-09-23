#!/usr/bin/env python
# coding: utf-8

# In[2]:


import gc
import os
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from PIL import Image
from sklearn.manifold import TSNE
from torchview import draw_graph
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPModel

import wandb

os.environ["HF_HOME"] = "../.hf_home"
random_seed = 42
torch.manual_seed(random_seed)
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


# In[2]:


model_vision = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336", cache_dir=os.environ['HF_HOME'])
model_text = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14-336", cache_dir=os.environ['HF_HOME'])
model_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14-336", cache_dir=os.environ['HF_HOME'])
model_image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336", cache_dir=os.environ['HF_HOME'])

clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336", cache_dir=os.environ['HF_HOME'])
vis_proj = clip.visual_projection
text_proj = clip.text_projection


# In[3]:


model_vision.eval()
model_text.eval()
print()


# In[4]:


model_graph = draw_graph(model_text, input_size=(1, 77), expand_nested=True, dtypes=[torch.long], device='cpu')
model_graph.visual_graph


# In[5]:


model_graph = draw_graph(model_vision, input_size=(1, 3, 336, 336), expand_nested=True, dtypes=[torch.float], device='cpu')
model_graph.visual_graph


# In[6]:


with torch.no_grad():
    tokenized = model_tokenizer.encode("a photo of a cat", return_tensors='pt')
print(tokenized.shape)
tokenized


# In[7]:


with torch.no_grad():
    encoded_text = model_text(tokenized)
print(encoded_text.pooler_output.shape)
encoded_text.pooler_output


# In[8]:


cat = Image.open("sample_images/cat.jpg").convert("RGB")
image = model_image_processor(images=cat, return_tensors="pt")
print(image['pixel_values'].shape)
image['pixel_values'][0,0]


# In[9]:


with torch.no_grad():
    encoded_image = model_vision(image['pixel_values'])
print(encoded_image.pooler_output.shape)
encoded_image.pooler_output


# In[10]:


proj_image = vis_proj(encoded_image.pooler_output)
proj_text = text_proj(encoded_text.pooler_output)
proj_image.shape, proj_text.shape


# In[11]:


data_folder = "./sample_images"
data = {}
for label in os.listdir(data_folder):
    if os.path.isdir(os.path.join(data_folder, label)):
        data[label] = []
        for file in os.listdir(os.path.join(data_folder, label)):
            if 'urls.txt' not in file:
                data[label].append(os.path.join(data_folder, label, file))

data_imgs = {}
for label in data:
    data_imgs[label] = []
    for img in data[label]:
        try:
            image = Image.open(img).convert("RGB")
            if image.size[0] < 100:
                del image
                continue
            data_imgs[label].append(image)
        except Exception as e:
            print(f"Error loading image {img}: {e}")


# In[12]:


def encode_images(images: List[Image.Image], batch_size=16, output_device="mps") -> torch.Tensor:
    all_embeddings = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        processed = model_image_processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            model_vision.to(device)
            vis_proj.to(device)
            encoded = model_vision(processed['pixel_values'])
            pooled = encoded.pooler_output
            projected = vis_proj(pooled)
        all_embeddings.append(projected.cpu())
        del processed, encoded, pooled, projected
        gc.collect()
    return torch.cat(all_embeddings, dim=0).to(output_device)

def encode_text(text: str, output_device="mps") -> torch.Tensor:
    model_text.eval()
    with torch.no_grad():
        model_text.to(device)
        text_proj.to(device)
        tokenized = model_tokenizer.encode(text, return_tensors='pt').to(device)
        encoded = model_text(tokenized)
        pooled = encoded.pooler_output
        projected = text_proj(pooled)

    del tokenized
    del encoded
    del pooled

    return projected.to(output_device)


# In[13]:


encoded = {}
for label, imgs in data_imgs.items():
    print(f"Encoding {len(imgs)} images for label {label}")
    encoded_images = encode_images(imgs, output_device="cpu")
    encoded_label = encode_text(label, output_device="cpu")
    encoded[label] = {
        "images": [{
            "image": img,
            "embedding": emb
        } for img, emb in zip(imgs, encoded_images)],
        "text_embedding": encoded_label.squeeze()
    }
    del encoded_images
    del encoded_label
    gc.collect()


# In[3]:


from io import BytesIO


def to_bytes(img: "Image.Image") -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def from_bytes(byte_im: bytes) -> "Image.Image":
    buf = BytesIO(byte_im)
    img = Image.open(buf)            # formats=("PNG",) optional
    img.load()                       # force read to avoid lazy I/O issues
    return img.convert("RGB")


# In[47]:


dfs = []
for label in encoded:
    text_emb = encoded[label]["text_embedding"].numpy()
    imgs, embs = zip(*[(item["image"], item["embedding"].numpy()) for item in encoded[label]["images"]])
    df = pd.DataFrame({
        "label": label,
        "image": imgs,
        "image_embedding": embs,
        "text_embedding": [text_emb] * len(imgs)
    })
    dfs.append(df)

df_concat = pd.concat(dfs, ignore_index=True)
df_concat['image'] = df_concat['image'].apply(lambda x: to_bytes(x))
df_concat.to_parquet("clip_image_text_embeddings.parquet", engine='fastparquet', index=False)


# In[4]:


df_concat = pd.read_parquet("clip_image_text_embeddings.parquet", engine='fastparquet')


# In[6]:


df_img = df_concat.drop(columns=["text_embedding"]).rename(columns={"image_embedding": "target"})
df_img['orig_img'] = df_img['image'].apply(lambda x: from_bytes(x))
df_img['image'] = df_img['image'].apply(lambda x: wandb.Image(from_bytes(x)))

df_txt = df_concat.drop(columns=["image", "image_embedding"]).rename(columns={"text_embedding": "target"}).drop_duplicates(subset="label")


# In[12]:


with wandb.init(project="clip-embeddings") as run:
    run.log({"image_embeddings": wandb.Table(dataframe=df_img), "text_embeddings": wandb.Table(dataframe=df_txt)})


# In[7]:


tnse = TSNE(n_components=3, random_state=random_seed, init='random', learning_rate='auto', perplexity=5)
results = tnse.fit_transform(np.array(df_img["target"].tolist() + df_txt["target"].tolist()))

img_proj = results[:len(df_img)]
txt_proj = results[len(df_img):]


# In[11]:


df_img['tnse-3d'] = img_proj.tolist()
df_img['tnse-3d-0'] = img_proj[:,0]
df_img['tnse-3d-1'] = img_proj[:,1]
df_img['tnse-3d-2'] = img_proj[:,2]

df_txt['tnse-3d'] = txt_proj.tolist()
df_txt['tnse-3d-0'] = txt_proj[:,0]
df_txt['tnse-3d-1'] = txt_proj[:,1]
df_txt['tnse-3d-2'] = txt_proj[:,2]


# In[12]:


# plot 3d with plotly, hover show image
fig = px.scatter_3d(df_img, x='tnse-3d-0', y='tnse-3d-1', z='tnse-3d-2', color='label', hover_data=['orig_img'], title="CLIP Image Embeddings with t-SNE")
fig.add_scatter3d(x=df_txt['tnse-3d-0'], y=df_txt['tnse-3d-1'], z=df_txt['tnse-3d-2'], mode='markers+text', marker=dict(size=8, symbol='diamond', color='black'), text=df_txt['label'], textposition='top center', name='Text Embeddings')
fig.update_traces(marker=dict(size=4), selector=dict(mode='markers'))
fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
fig.show()


# In[13]:


fig.write_html("clip_embeddings.html")


# In[ ]:




