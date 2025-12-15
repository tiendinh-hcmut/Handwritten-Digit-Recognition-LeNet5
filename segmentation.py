import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from improve import LeNet5

IMAGE_PATH = "images/test1.jpg"

def predit_line_of_digits(image_path, model, device, transform):
  global num
  img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
  img = cv2.bitwise_not(img)

  # just take the pixel has value from 128 to 255
  _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
  # use erosion to reduce the noise 
  thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, np.ones((5, 5), np.uint8))
  # use closing to connect components (small gap)
  thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
  plt.figure()
  plt.subplot(231)
  plt.imshow(img, cmap="gray")
  plt.subplot(232)
  plt.imshow(thresh, cmap="gray")
  
  # find contours (each segmentation) 
  contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cv2.drawContours(img, contours, -1, (0, 0, 255), 5)
  plt.subplot(233)
  plt.imshow(img, cmap="gray")

  # sort contours from left to right
  bounding_boxes = [cv2.boundingRect(c) for c in contours]
  bounding_boxes.sort(key=lambda x: x[0])
  print(bounding_boxes)
  full_number_string = ""

  for (x, y, w, h) in bounding_boxes:
    # ignore noise (if the contour below 5000 pixel)
    if w * h < 5000: 
      continue

    # crop into the digit
    pad = 5
    roi = thresh[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad]

    # convert it to pil to perform the resize and predict
    roi_pil = Image.fromarray(roi)
    digit_input = resize_and_pad(roi_pil)
    plt.subplot(234)
    plt.imshow(digit_input, cmap="gray")
    
    tensor_input = transform(digit_input).unsqueeze(0).to(device)
    with torch.no_grad():
      output = model(tensor_input)
      prediction = output.argmax(dim=1).item()
      full_number_string += str(prediction)
  
  return full_number_string

def resize_and_pad(img_pil):
  w, h = img_pil.size

  # convert it around 20 x 20
  ratio = min(20/w, 20/h)
  new_w = int(w * ratio)
  new_h = int(h * ratio)
  img_resized = img_pil.resize((new_w, new_h), Image.BILINEAR)

  # create the background
  final_img = Image.new("L", (28, 28), 0)
  
  # put the digit on the background
  paste_x = (28 - new_w) // 2
  paste_y = (28 - new_h) // 2
  final_img.paste(img_resized, (paste_x, paste_y))
  
  return final_img

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  try:
    model = LeNet5().to(device)
    model.load_state_dict(torch.load("lenet_improve.pth", map_location=device))
    model.eval()
    print(f"Loaded lenet_improve.pth")
  except Exception as e:
    print(f"Error loading lenet_improve.pth: {e}")
    pass

  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
  ])

  print(predit_line_of_digits(IMAGE_PATH, model, device, transform))
  plt.show()