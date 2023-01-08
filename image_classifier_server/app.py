from flask import Flask, jsonify, request
from PIL import Image

from inference import ImageClassifier

app = Flask("image classifier")


@app.route("/classify", methods=["POST"])
def classify():
    if request.files:
        img = request.files["image"]
        img = Image.open(img.stream)
        return jsonify(ImageClassifier.predict(img))

    return {"ok": False}


if __name__ == '__main__':
    app.run(debug=True)
