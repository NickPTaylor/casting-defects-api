import pathlib
import numpy as np
from keras.models import load_model
from keras_preprocessing.image import load_img, img_to_array

# Load model.
model = load_model(pathlib.Path.cwd() / 'model/full_model.h5')

def get_pred_from_file(f):
    img = load_img(f, color_mode='grayscale')
    img_arr = img_to_array(img)
    # IMPORTANT - apply same scaling as that applied in model.
    img_arr /= 255.
    img_arr = np.expand_dims(img_arr, axis=0)
    pred_prob = model.predict(img_arr)[0][0]

    return str(dict(probability=round(1 - pred_prob, 6)))
