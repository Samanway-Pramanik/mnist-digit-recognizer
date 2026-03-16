from flask import Flask, render_template, request, send_from_directory
import torch
import os

from model import CNN
from utils import preprocess_image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = CNN()
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=torch.device("cpu")))
model.eval()


@app.route("/", methods=["GET","POST"])
def index():

    prediction = None
    filename = None

    if request.method == "POST":

        file = request.files["image"]

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)

        file.save(filepath)

        filename = file.filename

        img = preprocess_image(filepath)

        with torch.no_grad():
            output = model(img)

            _, pred = torch.max(output,1)

        prediction = int(pred.item())

    return render_template("index.html", prediction=prediction, filename=filename)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)