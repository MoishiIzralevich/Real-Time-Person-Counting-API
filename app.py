import os
from fastapi import FastAPI, UploadFile, File
import tempfile
import shutil
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

# Load YOLO model
model = YOLO("models/yolov8n.pt")

# Load the pre-trained gender classifier (only used when requested)
faceProto = "models/opencv_face_detector.pbtxt"
faceModel = "models/opencv_face_detector_uint8.pb"
genderProto = "models/gender_deploy.prototxt"
genderModel = "models/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

@app.post("/process-video/")
async def process_video(file: UploadFile = File(...), use_gender_classification: bool = False, save_output: bool = False):
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_video_path = temp_file.name

    # Open the video using OpenCV
    cap = cv2.VideoCapture(temp_video_path)
    frame_index = 0
    unique_person_ids = set()
    male_count = set()
    female_count = set()

    # Get the video properties for output
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec
        output_video_path = "output_processed_video.mp4"
        out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Initialize the VideoWriter once we have the first frame (for consistent dimensions)
        if save_output and out is None:
            if out is None:
                frame_height, frame_width = frame.shape[:2]
                out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))

        # Detect persons using YOLO
        results = model.track(frame, conf=0.5, classes=[0], persist=True)  # Class 0 is 'person'

        for box in results[0].boxes:
            if box.cls == 0:  # Ensure it's a person
                person_id = int(box.id.item()) if box.id is not None else None

                if not person_id:
                    continue
                
                unique_person_ids.add(person_id)  # Add to global unique set

                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                person = frame[y1:y2, x1:x2]  # Crop the person

                if person.size == 0:
                    continue  # Skip empty or invalid person images

                # Draw the bounding box on the original frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if use_gender_classification:
                    # Detect face within the cropped person
                    blob = cv2.dnn.blobFromImage(person, 1.0, (300, 300), [104, 117, 123], True, False)
                    faceNet.setInput(blob)
                    detections = faceNet.forward()

                    for i in range(detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        if confidence > 0.7:
                            # Face detected; classify gender
                            face_x1 = max(0, int(detections[0, 0, i, 3] * person.shape[1]))
                            face_y1 = max(0, int(detections[0, 0, i, 4] * person.shape[0]))
                            face_x2 = min(person.shape[1] - 1, int(detections[0, 0, i, 5] * person.shape[1]))
                            face_y2 = min(person.shape[0] - 1, int(detections[0, 0, i, 6] * person.shape[0]))

                            face = person[face_y1:face_y2, face_x1:face_x2]
                            
                            if face.size == 0:
                                continue  # Skip empty or invalid face images

                            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

                            # Gender classification
                            genderNet.setInput(face_blob)
                            genderPreds = genderNet.forward()
                            gender = genderList[np.argmax(genderPreds[0])]

                            # Count male and female
                            if gender == 'Male':
                                male_count.add(int(box.id.item()))
                            else:
                                female_count.add(int(box.id.item()))

        if save_output:
            # Optionally add annotations on the frame
            cv2.putText(frame, f"Male: {len(male_count)} Female: {len(female_count)}  Total = {len(unique_person_ids)} ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Write the frame with annotations to the output video
            out.write(frame)

        # Print male and female counts for this frame
        print(f"Frame {frame_index + 1}: Male = {len(male_count)}, Female = {len(female_count)} Total = {len(unique_person_ids)}")

        frame_index += 1

    # Release the video capture and clean up
    cap.release()
    cv2.destroyAllWindows()
    os.remove(temp_video_path)
    if save_output:
        out.release()

    return {
        "message": "Video processed successfully." + (" Gender classification applied." if use_gender_classification else ""),
        "output_video_path": output_video_path,
        "total_unique_persons": len(unique_person_ids),
        "male_count": len(male_count),
        "female_count": len(female_count)
    }
