import os
import requests
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import clip
import json
import torch
import numpy as np
import faiss

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
IMAGE_FOLDER = "./train"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
with open("db/products.json", "r") as products:
    data = json.load(products)

# # Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load(
    "ViT-B/32", device=device)  # returns both model and preprocess

# Compute embeddings for all images
file_names = []
embeddings = []
index = None


def train():
    global file_names, embeddings, index
    # For training
    for fname in os.listdir(IMAGE_FOLDER):
        # Read all items names
        fpath = os.path.join(IMAGE_FOLDER, fname)
        image = Image.open(fpath)
        try:
            # preprocess to make image input compatible
            image_input = preprocess(image).unsqueeze(0).to(device)
            # make zero grad to remover overlapping on any calculations
            with torch.no_grad():
                # Get the image embedings with encoding
                image_emb = clip_model.encode_image(image_input)
            # convert embedings in vectors and flatten
            image_emb = image_emb.cpu().numpy().flatten()
            # Append image embed in list
            embeddings.append(image_emb)
            file_names.append(fname)
        except Exception as e:
            print(f"Skipping {fname}: {e}")

    embeddings = np.stack(embeddings).astype(np.float32)
    # print(embeddings.shape) # we are using 512 columns for each image

    # # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # # Save the index and file names
    faiss.write_index(index, "clip_images.index")

    with open("clip_images_files.json", "w") as f:
        for name in file_names:
            f.write(f"{name}\n")

    print("Index built and saved!")


# train feature
train()


# Search similar image
def search_image(query_img_path, top_k=3):
    global index, file_names
    # Open image
    query_image = Image.open(query_img_path)
    # check img mode = rgb or not
    if query_image.mode != "RGB":
        query_image = query_image.convert("RGB")

    # make Input format
    query_input = preprocess(query_image).unsqueeze(0).to(device)

    # Zero grad for remove other calculation with overlapping
    with torch.no_grad():
        # Encode image by AI modal
        query_emb = clip_model.encode_image(query_input)
        # print(f"(Encoded) query_emb.shape: {query_emb.shape}")

        # convert in float32 datatype
        query_emb = query_emb.cpu().numpy().astype(np.float32)
        # print(f"query_emb.shape: {query_emb.shape}")

        # we get distance and index
        D, I = index.search(query_emb, top_k)
        print(f"Top {top_k} similar images:")
        print(I)

        for rank, idx in enumerate(I[0]):
            print(f"{rank+1}: {file_names[idx]} with distance {D[0][rank]}")

        response = []
        for rank, idx in enumerate(I[0]):
            for i_num in range(len(data["params"])):
                if data["params"][i_num]["img"] == file_names[idx]:
                    response.append([{"Product_Img": data["params"][i_num]['img'],"Price": data["params"][i_num]['price'],"Link": data["params"][i_num]['link']}])
                    print("each data saved:",response)
                    
        if response:
            print("the data",response)
            return response,200
        return {"error": "No similar image found"}, 404


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Image Upload API",
        "endpoint": "/upload-image",
        "method": "POST",
        "body": {
            "image_url": "https://example.com/image.jpg"
        }
    }), 200


@app.route('/upload-image', methods=['POST', 'GET'])
def upload_imageprocess():
    image_url = request.args.get('image_url')

    if not image_url:
        data = request.get_json(silent=True)
        if data and 'image_url' in data:
            image_url = data['image_url']

    if not image_url:
        return {
            "error": "image_url is required (as query parameter or JSON body)"
        }, 400

    try:
        resp = requests.get(image_url, stream=True, timeout=30)
        if resp.status_code != 200:
            return {
                "error": f"Failed to download image, status {resp.status_code}"
            }, 400

        filename = image_url.split('/')[-1].split('?')[0]
        filename = secure_filename(filename) or "image.jpg"

        save_path = os.path.join(UPLOAD_FOLDER, filename)

        with open(save_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        response_data, status_code = search_image(save_path)
        print("main api in",response_data)
        if status_code == 200:
            os.remove(save_path)
        return jsonify(response_data), status_code

    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, ssl_context=(os.path.join("certificate.crt"),os.path.join("keyfile.key")))
    # app.run(host="0.0.0.0", port=5000)
