import io, traceback

from flask import Flask, request, g
from flask import send_file
from flask_mako import MakoTemplates, render_template
from plim import preprocessor

from PIL import Image, ExifTags
import numpy as np
import cv2

from skimage import transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def preprocess_img(image):
    label_3 = np.zeros(image.shape)
    label = np.zeros(label_3.shape[0:2])

    if (3 == len(label_3.shape)):
        label = label_3[:, :, 0]
    elif (2 == len(label_3.shape)):
        label = label_3

    if (3 == len(image.shape) and 2 == len(label.shape)):
        label = label[:, :, np.newaxis]
    elif (2 == len(image.shape) and 2 == len(label.shape)):
        image = image[:, :, np.newaxis]
        label = label[:, :, np.newaxis]

    transform = transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
    sample = transform({'imidx': np.array([0.]),'image': image, 'label': label})

    return sample

def run(img):
    torch.cuda.empty_cache()
    img_BGRA = None
    with torch.no_grad():
      sample = preprocess_img(img)
      inputs_test = sample['image'].unsqueeze(0)
      inputs_test = inputs_test.type(torch.FloatTensor)

      if torch.cuda.is_available():
          inputs_test = Variable(inputs_test.cuda())
      else:
          inputs_test = Variable(inputs_test)

      d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

      # normalization
      pred = d1[:,0,:,:]
      pred = normPRED(pred)

      predict = pred
      predict = predict.squeeze()
      predict_np = predict.cpu().data.numpy()

      im = Image.fromarray(predict_np*255).convert('RGB')
      imo = np.array(im.resize((img.shape[1],img.shape[0]),resample=Image.BILINEAR))

      del d1,d2,d3,d4,d5,d6,d7

      mask = np.greater(imo, imo.mean())
      crops = img*mask

      tmp = cv2.cvtColor(crops, cv2.COLOR_BGR2GRAY)
      ret, alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
      b_channel, g_channel, r_channel = cv2.split(crops)
      img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha))
    return Image.fromarray(img_BGRA)


app = Flask(__name__, instance_relative_config=True)

# For Plim templates
mako = MakoTemplates(app)
app.config['MAKO_PREPROCESSOR'] = preprocessor
app.config.from_object('config.ProductionConfig')

print("Loading model")
model_name='u2netp'#u2netp
model_dir = './saved_models/'+ model_name + '/' + model_name + '.pth'

if(model_name=='u2net'):
    print("...load U2NET---173.6 MB")
    net = U2NET(3,1)
elif(model_name=='u2netp'):
    print("...load U2NEP---4.7 MB")
    net = U2NETP(3,1)
net.load_state_dict(torch.load(model_dir,map_location=lambda storage, loc: storage))
if torch.cuda.is_available():
    net.cuda()
net.eval()

@app.route('/predict', methods=['POST'])
def predict():
    # Load image
    image = request.files['file']
    image = np.array(Image.open(image))
    img_BGRA = run(image)

    byte_io = io.BytesIO()
    img_BGRA.save(byte_io, 'PNG')
    byte_io.seek(0)
    return send_file(byte_io, mimetype='image/png')

@app.route('/')
def homepage():
    return render_template('index.html.slim', name='mako')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
