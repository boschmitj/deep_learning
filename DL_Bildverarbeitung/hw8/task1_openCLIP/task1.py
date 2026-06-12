import torch
from PIL import Image
import open_clip
import torchvision.datasets as ds
from torch.utils.data import Dataset, DataLoader 
import os   
from PIL import Image
import matplotlib.pyplot as plt 

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active


def load_images(path: str, preprocess):
    images = []
    images_preproccessed = []
    for file in os.listdir("./val2017"):
        filename = os.fsdecode(file)
        image = Image.open(filename).unsqueeze(0)
        images.append(image)
        images_preproccessed.append(preprocess(image))
    return images, images_preproccessed

def open_set_search(query : str, images, tokenizer, model):
    text = tokenizer([query])
    
    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        image_probs = (100. * image_features @ text_features.T).softmax(dim=-1)
    
    return torch.topk(image_probs, 10)

def plot_found_images(indices, images):
    _, axs = plt.subplots(2, 5, figsize=(15, 6))
    axs = axs.flatten()
    for idx in indices:
        image = images[idx]
        for ax in axs:
            ax.imshow(image)
        plt.show()

    

query = ""

images, images_preprocessed = load_images("./val2017", preprocess_val).to(device)
top_10_images_idx = open_set_search(query, images_preprocessed, tokenizer, model)[1]
plot_found_images(top_10_images_idx, images)

# image = preprocess_val(Image.open("docs/CLIP.png")).unsqueeze(0)
# text = tokenizer(["a diagram", "a dog", "a cat"])

# with torch.no_grad(), torch.autocast("cuda"):
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     text_features /= text_features.norm(dim=-1, keepdim=True)

#     text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]