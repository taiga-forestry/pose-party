import numpy as np

def flatten_joints(joints, joint_keys):
    flattened = []
    mask = []

    left_shoulder = joints.get("pose_11")
    right_shoulder = joints.get("pose_12")

    if left_shoulder and right_shoulder:
        center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
        center_y = (left_shoulder['y'] + right_shoulder['y']) / 2
    else:
        center_x, center_y = 0, 0

    for key in sorted(joint_keys):
        joint = joints.get(key)
        if joint is None:
            flattened.extend([0, 0])
            mask.append(0)
        else:
            shifted_x = joint['x'] - center_x
            shifted_y = joint['y'] - center_y
            flattened.extend([shifted_x, shifted_y])
            mask.append(1)

    return np.array(flattened, dtype=np.float32), np.array(mask, dtype=np.float32)

def cosine_distance(joints1, joints2):
    print(joints1.keys(), joints2.keys())
    joint_keys = set(joints1.keys()).intersection(joints2.keys())
    if not joint_keys:
        print("1")
        return 1.0

    pose1, mask1 = flatten_joints(joints1, joint_keys)
    pose2, mask2 = flatten_joints(joints2, joint_keys)

    # Apply mask to both poses
    # Apply mask to both poses
    mask = mask1 * mask2
    if np.sum(mask) == 0:
        print("2")
        return 1.0  # No valid joints

    # Expand mask to match the shape of the flattened pose arrays
    expanded_mask = np.repeat(mask, 2)

    pose1 *= expanded_mask
    pose2 *= expanded_mask

    norm1 = np.linalg.norm(pose1)
    norm2 = np.linalg.norm(pose2)

    if norm1 == 0 or norm2 == 0:
        print("3")
        return 1.0  # Avoid division by zero

    cosine_sim = np.dot(pose1, pose2) / (norm1 * norm2)
    return 1 - cosine_sim