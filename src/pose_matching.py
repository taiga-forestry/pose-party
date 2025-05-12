import numpy as np
from scipy.spatial.distance import cosine

# Chosen joints

SELECTED_JOINTS = [
    0,       # nose
    11, 12,  # shoulders
    13, 14,  # elbows
    15, 16,  # wrists
    23, 24,  # hips
    25, 26,  # knees
    27, 28,  # ankles
    29, 30,  # heels
    31, 32,  # feet
]

JOINT_WEIGHTS = {
    0: 1.0,     # nose
    11: 1.0, 12: 1.0,  # shoulders
    13: 5.0, 14: 5.0,  # elbows
    15: 7.0, 16: 7.0,  # wrists
    23: 1.0, 24: 1.0,  # hips
    25: 7.0, 26: 7.0,  # knees
    27: 7.0, 28: 7.0,  # ankles
    29: 5.0, 30: 5.0,  # heel
    31: 2.0, 32: 2.0,  # feet
}

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

        if l_shoulder[3] < 0.85 or r_shoulder[3] < 0.85:
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
        if joints1[i][3] > 0.85 or joints2[i][3] > 0.85
    ]

    if not valid_indices:
        return float("inf")  # or 0 score

    vec1 = np.array([joints1[i][:2] for i in valid_indices])
    vec2 = np.array([(joints2[i][0] - dx, joints2[i][1]) for i in valid_indices])
    weights = np.array([JOINT_WEIGHTS.get(i, 1.0) for i in valid_indices])

    distances = np.linalg.norm(vec1 - vec2, axis=1)
    mean_distance = np.average(distances, weights=weights)
    d = mean_distance
    d0 = 0.20
    k = 25
    score = 100 / (1 + np.exp(k * (d - d0)))

    # Compute cosine distance
    # distance = cosine(vec1, vec2)
    # similarities = [1 - cosine(v1, v2) for v1, v2 in zip(vec1, vec2)]
    # mean_similarity = np.mean(similarities)
    # score = max(0, 100 * (mean_similarity ** 10)) # max(0, 100 * mean_similarity)

    if not should_log:
        return score

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

        # f.write(f"similarity: {mean_similarity}\n")
        f.write(f"dist: {mean_distance}\n")
        f.write(f"score: {score}\n")
    
    return score