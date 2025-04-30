import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import math

# Model available to download here: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models
model_path = "models/pose_landmarker_lite.task"

video_source = 0

num_poses = 2
min_pose_detection_confidence = 0.5
min_pose_presence_confidence = 0.5
min_tracking_confidence = 0.5

# Cosine Distance Metric
# Input: L2 normized pose vectors
# Output: Cosine distance between the two vectors
def cosine_distance(pose1, pose2):
	# Find the cosine similarity
	cossim = pose1.dot(np.transpose(pose2)) / (np.linalg.norm(pose1) * np.linalg.norm(pose2))

	# Find the cosine distance
	cosdist = (1 - cossim)

	return cosdist

# Weighted Distance Metric
# Input: L2 normized pose vectors, and confidence scores for each point in pose 1
# Output: Weighted distance between the two vectors
def weight_distance(pose1, pose2, conf1):
	# D(U,V) = (1 / sum(conf1)) * sum(conf1 * ||pose1 - pose2||)
	#		 = sum1 * sum2

	# Compute first summation
	sum1 = 1 / np.sum(conf1)

	# Compute second summation
	sum2 = 0
	for i in range(len(pose1)):
		conf_ind = math.floor(i / 2) # each index i has x and y that share same confidence score
		sum2 += conf1[conf_ind] * abs(pose1[i] - pose2[i])

	weighted_dist = sum1 * sum2

	return weighted_dist

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z) for landmark in pose_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


def normalize_pose(pose_landmarks):
    # Returns a flat, L2-normalized vector of [x1, y1, x2, y2, ..., xN, yN]
    landmarks = np.array([[lmk.x, lmk.y] for lmk in pose_landmarks], dtype=np.float32)
    flat = landmarks.flatten()
    norm = np.linalg.norm(flat)
    return flat / norm if norm > 0 else flat

def extract_confidences(pose_landmarks):
    return np.array([lmk.visibility for lmk in pose_landmarks], dtype=np.float32)

def cosine_distance(pose1, pose2):
    cossim = pose1.dot(np.transpose(pose2)) / (np.linalg.norm(pose1) * np.linalg.norm(pose2))
    return 1 - cossim

# Weighted Distance Metric
# Input: L2-normalized pose vectors, and confidence scores for each point in pose1
# Output: Weighted distance between the two vectors
def weight_distance(pose1, pose2, conf1):
    sum1 = 1 / np.sum(conf1)
    sum2 = 0
    for i in range(len(pose1)):
        conf_ind = math.floor(i / 2)  # x/y share same visibility
        sum2 += conf1[conf_ind] * abs(pose1[i] - pose2[i])
    return sum1 * sum2

to_window = None
last_timestamp_ms = 0


def print_result(detection_result: vision.PoseLandmarkerResult, output_image: mp.Image,
                 timestamp_ms: int):
    global to_window, last_timestamp_ms
    if timestamp_ms < last_timestamp_ms:
        return
    last_timestamp_ms = timestamp_ms

    # Draw landmarks
    image_np = output_image.numpy_view()
    to_window = cv2.cvtColor(draw_landmarks_on_image(image_np, detection_result), cv2.COLOR_RGB2BGR)

    # Compare if there are 2 poses detected
    if len(detection_result.pose_landmarks) >= 2:
        pose1 = detection_result.pose_landmarks[0]
        pose2 = detection_result.pose_landmarks[1]

        norm_pose1 = normalize_pose(pose1)
        norm_pose2 = normalize_pose(pose2)
        conf1 = extract_confidences(pose1)

        cos_dist = cosine_distance(norm_pose1, norm_pose2)
        weighted_dist = weight_distance(norm_pose1, norm_pose2, conf1)

        print(f"Cosine Distance: {cos_dist:.4f}, Weighted Distance: {weighted_dist:.4f}")

        # Optional: overlay on frame
        cv2.putText(to_window, f"CosDist: {cos_dist:.3f}, WDist: {weighted_dist:.3f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)



base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_poses=num_poses,
    # min_pose_detection_confidence=min_pose_detection_confidence,
    # min_pose_presence_confidence=min_pose_presence_confidence,
    # min_tracking_confidence=min_tracking_confidence,
    # output_segmentation_masks=False,
    result_callback=print_result
)

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    # Use OpenCV’s VideoCapture to start capturing from the webcam.
    cap = cv2.VideoCapture(video_source)

    # Create a loop to read the latest frame from the camera using VideoCapture#read()
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Image capture failed.")
            break

        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)

        if to_window is not None:
            cv2.imshow("MediaPipe Pose Landmark", to_window)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()