import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from collections import defaultdict
from game import DuelGameState

mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_landmarker = None

SELECTED_JOINTS = set([
    0,       # nose
    11, 12,  # shoulders
    13, 14,  # elbows
    15, 16,  # wrists
    23, 24,  # hips
    25, 26,  # knees
    28, 29,  # ankles
    29, 30,  # heels
    31, 32,  # feet
])

def initialize_landmarker():
    global mp_landmarker

    MODEL_PATH = "models/pose_landmarker_full.task"
    NUM_POSES = 2

    if mp_landmarker:
        return mp_landmarker

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

    mp_landmarker = vision.PoseLandmarker.create_from_options(options)
    return mp_landmarker

# documentation: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python
def get_and_draw_joints(landmarker, frame, game_state:DuelGameState):
    # convert given frame to RGB mp.Image for joint detection w/ landmarker
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect(mp_image)
    player_joints = {}

    # loop over joints for each person detected in frame
    for i, joints_list in enumerate(result.pose_landmarks):
        normalized_joints = [landmark_pb2.NormalizedLandmark(x=joint.x, y=joint.y, z=joint.z) for joint in joints_list]
        joints_protobuf = landmark_pb2.NormalizedLandmarkList(landmark=normalized_joints)
        mp_draw.draw_landmarks(frame, joints_protobuf, mp_pose.POSE_CONNECTIONS)

        player_joints[i] = [(joint.x, joint.y, joint.z, joint.visibility) for joint in joints_list]
    
    return player_joints

def get_and_save_joints(landmarker, frame, game_state:DuelGameState):
    h, w, _ = frame.shape
    # convert given frame to RGB mp.Image for joint detection w/ landmarker
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect(mp_image)

    player_id = game_state.curr_player.id - 1

    print("player_id", player_id, len(result.pose_landmarks))

    # loop over joints for each person detected in frame
    for i, joints_list in enumerate(result.pose_landmarks):
        if i != player_id and len(result.pose_landmarks) > 1:
            continue

        print("ASNLNALDNSLDNA")

        for j, landmark in enumerate(joints_list):
            if j not in SELECTED_JOINTS:
                continue
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            visibility = landmark.visibility
            game_state.saved_frame[player_id][j] = {"x": x, "y": y, "visibility": visibility}
