from flask import Flask, request, render_template

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
    return render_template('app.js', image_path=image_path)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
