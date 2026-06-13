import PIL.Image
from torchvision.utils import draw_bounding_boxes

from torchvision.transforms.functional import to_pil_image
import torchvision.models.detection as detection
import torch
import os

def save_prediction(image_id, output, save_dir):
  boxes = output["boxes"]
  scores = output["scores"]
  labels = output["labels"]

  keep = scores > 0.5
  
  os.makedirs(save_dir, exist_ok=True)
  
  with open(f"{save_dir}/{image_id}.txt", "w") as f:
    for i in range(len(scores[keep])):
      cls = labels[keep][i].item()
      score = scores[keep][i].item()
      x1, y1, x2, y2 = boxes[keep][i].tolist()
      
      f.write(f"{cls} {score:.4f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}")
      

def run():
  # hw8/codeforstudents/testimages1

  filename = './hw8/codeforstudents/testimages1/cows-5080091_960_720.jpg'
  path = "./hw8/val2017/"
  #https://pytorch.org/vision/stable/models.html  
  modelweights = detection.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1

  model = detection.retinanet_resnet50_fpn_v2(weights=modelweights, progress=True)  # your model, should use modelweights
  
  transforms = modelweights.transforms()
  model.eval()

  for file in os.listdir(path=path):
    img_path = path + "/" + file
    
    # prepare batch made from one image
    img = PIL.Image.open(filename).convert('RGB')  
    batch = [transforms(img)]
    
    # if you forget that, you will see a nice error :) 
    
    with torch.no_grad():
      # call your model on the batch
      predictions = model(batch)
      
      predicted = predictions[0] # batch has size 1, see above, so we grab that one element here

      # get names of labels which are predicted 
      labels_in_image = [modelweights.meta["categories"][i] for i in predicted["labels"]]
      print(labels_in_image)
      
      save_prediction(image_id=str.split(file, ".")[0], output=predicted, save_dir="coco_predictions")

if __name__=='__main__':
  run()  


    