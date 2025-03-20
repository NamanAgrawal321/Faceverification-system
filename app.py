from flask import Flask, request, jsonify
import numpy as np
import cv2
from deepface import DeepFace

app = Flask(__name__)

@app.route('/verify', methods=['POST','GET'])
def verify_faces():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({"error": "Both image1 and image2 are required"}), 400

    image1 = request.files['image1'].read()
    image2 = request.files['image2'].read()

    npimg1 = np.frombuffer(image1, np.uint8)
    npimg2 = np.frombuffer(image2, np.uint8)
    img1 = cv2.imdecode(npimg1, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(npimg2, cv2.IMREAD_COLOR)

    try:
        result = DeepFace.verify(img1_path=img1, img2_path=img2, enforce_detection=False)
        return jsonify({
            "distance": result["distance"],
            "similarity_score": (1 - result["distance"]) * 100,
            "verified": result["verified"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
