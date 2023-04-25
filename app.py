from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import tensorflow as tf

model = tf.keras.models.load_model("breast_model.h5")
#model.load_weights('mymodel.h5')


app = Flask(__name__)

dic={0 :"IDC Negative", 1 : "IDC Positive"}

#prediction
def predict_label(path):
    img=Image.open(path).convert('RGB')
    img=img.resize((50,50),resample=Image.BILINEAR)
    img=np.array(img,dtype=np.float32)/255.0
    test_img=np.expand_dims(img,axis=0)

    pre=model.predict(test_img)
    return dic[np.argmax(pre)]
    #pre_class=np.argmax(pre)

    
# Define a route for your homepage
@app.route('/', methods=['GET'])
def hello_world():
    return {"members": ["mem1","mem2"]}


# Define a route for uploading an image
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)
        pred = predict_label(img_path)
        

##    if pre_class==1:
##        print("IDC Positive")
##    else:
##        print("IDC Negative")
        
    return render_template('app.js',prediction = pred,image_path=img_path)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
