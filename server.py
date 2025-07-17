from flask import Flask, render_template, request, jsonify
import io, numpy as np
from search import FAISS
import os

def create_app():
    app = Flask(__name__)
    faiss = FAISS()

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/get_person", methods=["POST"])
    def get_person():
        raw_bytes   = request.data
        float_array = np.frombuffer(raw_bytes, dtype=np.float32)
        result = faiss.search(float_array)
        return jsonify({"status":"ok","output":result})

    @app.route("/add_person", methods=["POST"])
    def add_person():
        # Expecting multipart form: 'name' and 'audio' file
        if 'name' not in request.form or 'audio' not in request.files:
            return jsonify({"status":"error","msg":"name or audio missing"}), 400

        name = request.form['name']
        audio_file = request.files['audio']
        raw_bytes = audio_file.read()
        float_array = np.frombuffer(raw_bytes, dtype=np.float32)

        # Compute embedding and add to FAISS
        faiss.add_new_member(name, float_array) 

        return jsonify({"status":"ok","msg":f"Added speaker '{name}'"})

    return app

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    app = create_app()
    app.run(debug=True)
