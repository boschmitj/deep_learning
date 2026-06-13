import PIL.Image
from torchvision.utils import draw_bounding_boxes

from torchvision.transforms.functional import to_pil_image
import torchvision.models.detection as detection
import torch



def run():
  # hw8/codeforstudents/testimages1

  filename = './hw8/codeforstudents/testimages1/cows-5080091_960_720.jpg'

  #https://pytorch.org/vision/stable/models.html  
  modelweights = detection.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1

  model = detection.retinanet_resnet50_fpn_v2(weights=modelweights, progress=True)  # your model, should use modelweights
  
  transforms = modelweights.transforms()
  
  # prepare batch made from one image
  img = PIL.Image.open(filename).convert('RGB')  
  batch = [transforms(img)]
  
  # if you forget that, you will see a nice error :) 
  model.eval()
  
  with torch.no_grad():

    # call your model on the batch
    predictions = model(batch)
    
    predicted = predictions[0] # batch has size 1, see above, so we grab that one element here
    high_confidence = predicted["scores"] > 0.4

    #get names of labels which are predicted 
    labels_in_image = [modelweights.meta["categories"][i] for i in predicted["labels"]]
    print(labels_in_image)
    labels_in_image_high_confidence = [modelweights.meta["categories"][i] for i in predicted["labels"][high_confidence]]

  # output what is predicted to understand the format of torchvision model outputs
  # it is a list of dictionaries. list has length equal to batchsize
  print(type(predictions))
  print(type(predicted))
  # each dictionary has keys for the outputs which one would expect from an object detection neural net
  for k in predicted.keys():
    print('dictionary key name {}'.format(k))  

  print(predicted)    
  print('unique predicted labels (no confidence threshold):')
  print( set(labels_in_image))
  print('unique predicted labels (confidence threshold: 0.4):')
  print(set(labels_in_image_high_confidence))

  
  

  # draw the boundingbox (ALL)
  box = draw_bounding_boxes(batch[0], boxes=predicted["boxes"],
                            labels=labels_in_image,
                            colors="red",
                            width=4 ) #, font_size=30, font =None)
  #print convert to a PIL.Image                                                      
  im = to_pil_image(box.detach())
  #display the PIL.Image
  im.save("prediciton_result.jpg")
  
  # draw boundingbox (ONLY HIGH CONFIDENCE)
  box = draw_bounding_boxes(batch[0], boxes=predicted["boxes"][high_confidence],
                            labels=labels_in_image_high_confidence,
                            colors="blue",
                            width=4)
  im = to_pil_image(box.detach())
  im.save("prediction_result_high_confidence.jpg")



if __name__=='__main__':
  run()  


    