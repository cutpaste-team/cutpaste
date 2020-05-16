import io, traceback

from flask import Flask, request, g
from flask import send_file
from flask_mako import MakoTemplates, render_template
from plim import preprocessor

from PIL import Image, ExifTags
import numpy as np
import tensorflow as tf
import cv2

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  MODEL_DIR = "./model/frozen_inference_graph.pb"
  def __init__(self):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    with tf.gfile.GFile(self.MODEL_DIR, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map

app = Flask(__name__, instance_relative_config=True)

# For Plim templates
mako = MakoTemplates(app)
app.config['MAKO_PREPROCESSOR'] = preprocessor
app.config.from_object('config.ProductionConfig')

print("Loading model")
model = DeepLabModel()

@app.route('/predict', methods=['POST'])
def predict():
    # Load image
    image = request.files['file']
    image = Image.open(image)
    resized_im, seg_map = model.run(image)

    mask = np.greater(seg_map, 0) # get only non-zero positive pixels/labels
    mask = np.expand_dims(mask, axis=-1) # (H, W) -> (H, W, 1)
    mask = np.concatenate((mask, mask, mask), axis=-1) # (H, W, 1) -> (H, W, 3), (don't like it, so if you know how to do it better, please let me know)
    crops = resized_im * mask # apply mask on image

    tmp = cv2.cvtColor(crops, cv2.COLOR_BGR2GRAY)
    ret, alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b_channel, g_channel, r_channel = cv2.split(crops)
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha))
    img_BGRA = Image.fromarray(img_BGRA)
    byte_io = io.BytesIO()
    img_BGRA.save(byte_io, 'PNG')
    byte_io.seek(0)
    return send_file(byte_io, mimetype='image/png')

@app.route('/')
def homepage():
    return render_template('index.html.slim', name='mako')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
