import torch
from PIL import Image
import open_clip
import torchvision.datasets as ds
from torch.utils.data import Dataset, DataLoader 
import os   
from PIL import Image
import matplotlib.pyplot as plt 
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
model.to(device)
tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active



def load_images(path: str, preprocess):
    images = []
    images_preproccessed = []
    for file in os.listdir(path):
        filename = path + "/" + os.fsdecode(file) 
        image = Image.open(filename)
        images.append(image)
        images_preproccessed.append(preprocess(image))
    return images, images_preproccessed

def open_set_search(query : str, loader, tokenizer, model):
    text = tokenizer([query]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    image_probs = []
    for images in loader:
        images = images.to(device)
        with torch.no_grad():
            region_batches = augment(images)
            region_scores = []
            for region_images in region_batches:
                image_features = model.encode_image(region_images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                region_scores.append((image_features @ text_features.T).squeeze(1))

            image_probs_batch = torch.stack(region_scores, dim=0).max(dim=0).values
            print(image_probs_batch.shape)
            image_probs.append(image_probs_batch)

    image_probs = torch.cat(image_probs)
    return torch.topk(image_probs, 10)

def plot_found_images(indices, images):
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    axs = axs.flatten()
    for ax, idx in zip(axs, indices):
        ax.imshow(images[idx])
        ax.axis("off")
    plt.show()

def augment(images):
    _, _, height, width = images.shape
    half_height = height // 2
    half_width = width // 2

    return [
        images,
        F.interpolate(images[:, :, :half_height, :half_width], size=(height, width), mode="bilinear", align_corners=False),
        F.interpolate(images[:, :, :half_height, half_width:], size=(height, width), mode="bilinear", align_corners=False),
        F.interpolate(images[:, :, half_height:, :half_width], size=(height, width), mode="bilinear", align_corners=False),
        F.interpolate(images[:, :, half_height:, half_width:], size=(height, width), mode="bilinear", align_corners=False),
    ]

query = "a cat"

images, images_preprocessed = load_images("../val2017", preprocess_val)
print(type(images_preprocessed[0]))

dataset = torch.stack(images_preprocessed)
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False
)

top_10_images_idx = open_set_search(query, loader, tokenizer, model)[1]
plot_found_images(top_10_images_idx, images)