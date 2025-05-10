import numpy as np
from scipy.spatial.distance import cosine

# Chosen joints

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

CONFIDENCE_THRESHOLD = 0.8

def calculate_similarity(game_state, should_log=False):
    joints1, joints2 = game_state.player_1.saved_joints, game_state.player_2.saved_joints

    # Filter visible and selected joints
    # def preprocess(joints):
    #     # Use midpoint between shoulders (11, 12) as origin
    #     s1, s2 = joints[11], joints[12]
    #     if s1[3] < 0.5 or s2[3] < 0.5:
    #         return None  # Shoulders not visible, can't normalize

    #     origin = np.array([(s1[0] + s2[0]) / 2, (s1[1] + s2[1]) / 2, (s1[2] + s2[2]) / 2])
    #     vec = []

    #     for i in SELECTED_JOINTS:
    #         x, y, z, v = joints[i]
    #         if v < 0.5:
    #             continue  # skip invisible joints
    #         joint = np.array([x, y, z])
    #         vec.append(joint - origin)  # normalize by shoulder midpoint

    #     return vec

    # vec1 = preprocess(joints1)
    # vec2 = preprocess(joints2)

    # if vec1 is None or vec2 is None:
    #     return float("inf")  # shoulders missing, can't compare

    # # Match vector lengths by truncating to shared visible joints
    # length = min(len(vec1), len(vec2))
    # if length == 0:
    #     return float("inf")  # no common joints to compare

    # vec1 = np.concatenate(vec1[:length])
    # vec2 = np.concatenate(vec2[:length])

    def find_center(joints):
        l_shoulder, r_shoulder = joints[11], joints[12]

        if l_shoulder[3] < CONFIDENCE_THRESHOLD or r_shoulder[3] < CONFIDENCE_THRESHOLD:
            raise ValueError("wat")
        
        center_x, center_y = (l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2
        return center_x, center_y

    p1_center = find_center(joints1)
    p2_center = find_center(joints2)
    dx = p2_center[0] - p1_center[0]
    # dy = p2_center - p1_center

    # joints1 = [(x, y) for (x,y,_,_) in joints1]
    # joints2 = [(x - dx, y) for (x,y,_,_) in joints2]

    valid_indices = [
        i for i in SELECTED_JOINTS
        if joints1[i][3] > CONFIDENCE_THRESHOLD and joints2[i][3] > CONFIDENCE_THRESHOLD
    ]

    if not valid_indices:
        return float("inf")  # or 0 score

    vec1 = np.array([joints1[i][:2] for i in valid_indices])
    vec2 = np.array([(joints2[i][0] - dx, joints2[i][1]) for i in valid_indices])

    # Compute cosine distance
    # distance = cosine(vec1, vec2)
    similarities = [1 - cosine(v1, v2) for v1, v2 in zip(vec1, vec2)]
    mean_similarity = np.mean(similarities)
    score = max(0, 100 * (mean_similarity ** 10)) # max(0, 100 * mean_similarity)

    if not should_log:
        return score
    
    if len(valid_indices) < 5:
        score -= 50

    with open(f"scores-{game_state.round}-{game_state.t}.txt", "w") as f:
        f.write(f"center1: {p1_center}\n")
        f.write(f"center2: {p2_center}\n")
        f.write(f"delta: {dx}\n")


        f.write(f"valid indices: {valid_indices}\n")

        f.write("player 1:\n")
        for x,y in vec1:
            f.write(f"({x}, {y})\n")

        f.write("player 2:\n")
        for x,y in vec2:
            f.write(f"({x}, {y})\n")

        f.write(f"similarity: {mean_similarity}\n")
        f.write(f"score: {score}\n")
    
    return score


# import numpy as np

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

# def calculate_similarity(joints1, joints2):
#     joints1 = [j for j in joints1]
#     joints1 = [j for j in joints2]

    

# def flatten_joints(joints, joint_keys):
#     flattened = []
#     mask = []

#     # Use shoulders as the center
#     left_shoulder = joints.get("pose_11")
#     right_shoulder = joints.get("pose_12")

#     if left_shoulder and right_shoulder:
#         center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
#         center_y = (left_shoulder['y'] + right_shoulder['y']) / 2
#     else:
#         center_x, center_y = 0, 0

#     for key in sorted(joint_keys):
#         joint = joints.get(key)
#         if joint is None:
#             flattened.extend([0, 0])
#             mask.append(0)
#         else:
#             shifted_x = joint['x'] - center_x
#             shifted_y = joint['y'] - center_y
#             flattened.extend([shifted_x, shifted_y])
#             mask.append(1)

#     return np.array(flattened, dtype=np.float32), np.array(mask, dtype=np.float32)

# def cosine_distance(joints1, joints2):
#     joint_keys = set(joints1.keys()).intersection(joints2.keys())
#     if not joint_keys:
#         print("1")
#         return 1.0  # No common joints

#     pose1, mask1 = flatten_joints(joints1, joint_keys)
#     pose2, mask2 = flatten_joints(joints2, joint_keys)

#     # Apply mask to both poses
#     # Apply mask to both poses
#     mask = mask1 * mask2
#     if np.sum(mask) == 0:
#         print("2")
#         return 1.0  # No valid joints

#     # Expand mask to match the shape of the flattened pose arrays
#     expanded_mask = np.repeat(mask, 2)

#     pose1 *= expanded_mask
#     pose2 *= expanded_mask

#     norm1 = np.linalg.norm(pose1)
#     norm2 = np.linalg.norm(pose2)

#     if norm1 == 0 or norm2 == 0:
#         print("3")
#         return 1.0  # Avoid division by zero

#     cosine_sim = np.dot(pose1, pose2) / (norm1 * norm2)
#     return 1 - cosine_sim

# def distance_to_percentage(distance):
#     return (1 - distance) * 100