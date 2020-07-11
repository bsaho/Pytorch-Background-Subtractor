from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import glob
import os

#######################################Segmentation program############################################333
torch.cuda.is_available()
device=torch.device ('cuda')

print (torch.cuda.is_available())

# Apply the transformations needed
import torchvision.transforms as T



# Define the helper function
def decode_segmap(image, source, nc=21):
  
  label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (255, 255, 255), (255, 255, 255), (255,255,255), (255,255,255),(255,255,255),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (255,255,255),(255,255,255), (255,255,255), (255,255,255),(255,255,255),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
              (255,255,255), (255,255,255), (255,255,255), (255,255,255),(255,255,255),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
              (255,255,255), (255,255,255), (255,255,255),(255,255,255), (255,255,255)])

  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  

  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
    
    rgb = np.stack([r, g, b], axis=2)

  # Load the foreground input image 
  foreground = cv2.imread(source)

  # Change the color of foreground image to RGB 
  # and resize image to match shape of R-band in RGB output map
#   foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
  foreground = cv2.resize(foreground,(r.shape[1],r.shape[0]))

  # Create a background array to hold white pixels
  # with the same size as RGB output map
  background = 0 * np.ones_like(rgb).astype(np.uint8)

  # Convert uint8 to float
  foreground = foreground.astype(float)
  background = background.astype(float)

  # Create a binary mask of the RGB output map using the threshold value 0
  th, alpha = cv2.threshold(np.array(rgb),0,255, cv2.THRESH_BINARY)
  

  # Apply a slight blur to the mask to soften edges
  alpha = cv2.GaussianBlur(alpha, (7,7),0)

  # Normalize the alpha mask to keep intensity between 0 and 1
  alpha = alpha.astype(float)/255
  return alpha

  # Multiply the foreground with the alpha matte
  foreground = cv2.multiply(alpha, foreground)  
  
  # Multiply the background with ( 1 - alpha )
  background = cv2.multiply(1.0 - alpha, background)  
  
  # Add the masked foreground and background
  outImage = cv2.add(foreground, background)

  # Return a normalized output image for display
  return outImage/255


#passes the image to deeplab for segmentation
def segment(net, path, folderName, show_orig=False, dev='cuda'):
    torch.cuda.empty_cache()
    if device.type == 'cuda':
      img = Image.open(path)
      if show_orig: plt.imshow(img); plt.axis('off'); plt.show()
      # Comment the Resize and CenterCrop for better inference results
      trf = T.Compose([T.Resize(450), 
                      #T.CenterCrop(224), 
                      T.ToTensor(), 
                      T.Normalize(mean = [0.485, 0.456, 0.406], 
                                  std = [0.229, 0.224, 0.225])])
      inp = trf(img).unsqueeze(0).to(device)
      out = net.to(device)(inp)['out']
      out.cuda ()
      om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()

      rgb = decode_segmap(om, path)
      imgName=path.split ("/")
      imgName= imgName[-1]
      dirrString="C:/Users/Desktop/modified_dataset/" + folderName  +"/Results"
      try:

        os.mkdir (dirrString)
      except:
        pass
      imgPath=  "C:/Users/Desktop/modified_dataset/" + folderName  + "/Results/" + imgName[:len(imgName)-4]  + ".png"

      print (imgPath)
      try:
        plt.imshow(rgb);plt.axis('off') ;plt.show()
        # plt.savefig (imgPath) 
      except:
        pass


dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()
txtfiles= os.listdir ('C:/Users/Desktop/modified_dataset/')
# print (txtfiles)

torch.cuda.empty_cache()


#runs through entire program and segments
for i in txtfiles:
  dirString = 'C:/Users/Desktop/modified_dataset/' + i +  "/input"
  # print (dirString)
  fileNames= os.listdir (dirString)

  for j in fileNames:
    imgString= dirString +"/" +j
    segment(dlab, imgString, i)