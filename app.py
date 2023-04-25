from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import tensorflow as tf

model = tf.keras.models.load_model("breast_model.h5")
#model.load_weights('mymodel.h5')
print("model load done")
app = Flask(__name__)

# Define a route for your homepage
@app.route('/', methods=['GET'])
def hello_world():
    return {"members": ["mem1","mem2"]}

# Define a route for uploading an image
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image from the request
    imagefile = request.files['imagefile']
    # Save the image to a file on your server
    image_path = './images/' + imagefile.filename
    imagefile.save(image_path)
    # Render the app.js file to display the uploaded image

    img=Image.open(image_path).convert('RGB')
    img=img.resize((50,50),resample=Image.BILINEAR)
    img=np.array(img,dtype=np.float32)/255.0
    test_img=np.expand_dims(img,axis=0)

    pre=model.predict(test_img)
    pre_class=np.argmax(pre)

    if pre_class==1:
        print("IDC Positive")
    else:
        print("IDC Negative")
    #prediction
    pre=model.predict(np.expend_dims(img,axis=0))[0]
    print("predict result : ",pre)
    return render_template('app.js', image_path=pre)#image_path=image_path)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
