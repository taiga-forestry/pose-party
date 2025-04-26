import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# SELECTED_JOINTS = set([
#     0,       # nose
#     11, 12,  # shoulders
#     13, 14,  # elbows
#     15, 16,  # wrists
#     23, 24,  # hips
#     25, 26,  # knees
#     28, 29,  # ankles
#     29, 30,  # heels
#     31, 32,  # feet
# ])

def initialize_landmarker():
    MODEL_PATH = "models/pose_landmarker_lite.task"
    NUM_POSES = 2

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=NUM_POSES,
        # min_pose_detection_confidence=min_pose_detection_confidence,
        # min_pose_presence_confidence=min_pose_presence_confidence,
        # min_tracking_confidence=min_tracking_confidence,
        # output_segmentation_masks=False,
        # running_mode=vision.RunningMode.LIVE_STREAM,
        # result_callback=print_result
    )

    return vision.PoseLandmarker.create_from_options(options)

# documentation: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python
def get_and_draw_joints(landmarker, frame):
    # convert given frame to RGB mp.Image for joint detection w/ landmarker
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect(mp_image)

    # loop over joints for each person detected in frame
    for joints_list in result.pose_landmarks:
        normalized_joints = [landmark_pb2.NormalizedLandmark(x=joint.x, y=joint.y, z=joint.z) for joint in joints_list]
        joints_protobuf = landmark_pb2.NormalizedLandmarkList(landmark=normalized_joints)
        mp_draw.draw_landmarks(frame, joints_protobuf, mp_pose.POSE_CONNECTIONS)