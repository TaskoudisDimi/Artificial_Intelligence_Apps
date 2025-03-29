from flask import Blueprint, render_template, request, jsonify

activity_recognition_bp = Blueprint('activity_recognition', __name__)

@activity_recognition_bp.route('/Activity_Recognition', methods=['GET'])
def activity_recognition():
    return render_template('Activity_Recognition.html')

@activity_recognition_bp.route('/Predict_Activity_Recognition', methods=['POST'])
def predict_activity_recognition():
    if request.method == "POST":
        uploaded_video = request.files['video']
        if uploaded_video is not None:
            # Add logic to process the uploaded video and make predictions
            print("Received the video for activity recognition.")
            # prediction_result = predict_activity(uploaded_video)
            return jsonify({'message': 'Video processed successfully.'})
        else:
            return jsonify({'message': 'Error: No video uploaded.'}), 400