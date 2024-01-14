from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import StringLookup
from tensorflow import keras
import onnxruntime
import os
from PIL import Image
from io import BytesIO

app = Flask(__name__)

onnx_model_path = 'model.onnx.com'
ort_session = onnxruntime.InferenceSession(onnx_model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get input data from the form
            input_data = request.files['input_image']

            def allowed_file(filename):
                return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}
            if input_data and allowed_file(input_data.filename):
                img = Image.open(BytesIO(input_data.read()))
                img = img.convert('L', palette=Image.ADAPTIVE, colors=256)
                file_path = 'app/static/tempimg.png'
                img.save(file_path, 'PNG')

            characters = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
            char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)
            num_to_char = StringLookup(
                vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
            )

            def distortion_free_resize(image, img_size):
                w, h = img_size
                image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

                pad_height = h - tf.shape(image)[0]
                pad_width = w - tf.shape(image)[1]

                if pad_height % 2 != 0:
                    height = pad_height // 2
                    pad_height_top = height + 1
                    pad_height_bottom = height
                else:
                    pad_height_top = pad_height_bottom = pad_height // 2

                if pad_width % 2 != 0:
                    width = pad_width // 2
                    pad_width_left = width + 1
                    pad_width_right = width
                else:
                    pad_width_left = pad_width_right = pad_width // 2

                image = tf.pad(
                    image,
                    paddings=[
                        [pad_height_top, pad_height_bottom],
                        [pad_width_left, pad_width_right],
                        [0, 0],
                    ],
                )

                image = tf.transpose(image, perm=[1, 0, 2])
                image = tf.image.flip_left_right(image)
                return image
            
            image_width = 128
            image_height = 32


            def preprocess_image(image_path, img_size=(image_width, image_height)):
                image = tf.io.read_file(image_path)
                image = tf.image.decode_png(image, 1)
                image = distortion_free_resize(image, img_size)
                image = tf.cast(image, tf.float32) / 255.0
                return image
            
            def decode_batch_predictions(pred):
                max_len = 21
                input_len = np.ones(pred.shape[0]) * pred.shape[1]
                results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
                    :, :max_len
                ]
                output_text = []
                for res in results:
                    res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
                    res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
                    output_text.append(res)
                return output_text

            input_name = ort_session.get_inputs()[0].name
            output_name = ort_session.get_outputs()[0].name

            input_image = preprocess_image(file_path)
            input_data = np.expand_dims(input_image, axis=0)

            ort_result = ort_session.run([output_name], {input_name: input_data})

            decoded_result = decode_batch_predictions(ort_result[0])

            result = f"Prediction: '{decoded_result[0]}'"

            return render_template('index.html', result=result)
        except Exception as e:
            error = f"An error occurred: {e}"
            return render_template('index.html', error=error)

if __name__ == '__main__':
    app.run(debug=True)
