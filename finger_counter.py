#This library is used to read terminal arguments, in the code if the user wants to increse the max hand count or the number of cameras, we also enter an argument.
import argparse #Example: python finger_counter.py --max-hands 8 --camera 1
#Math library is used to calculate distances, angles and hand rotation.
import math
from collections import deque #Used to store the last few detected finger counts.
#This is stored since real time computer vision can flicker especially when the hand is sideways.
#Its used as a smoothing technique.

import cv2 #One of the main libraries, used for camera and frame processes.
import mediapipe as mp #The main library used, a pre-trained computer vision framework that is used foe hand landmark detection.


DEFAULT_MAX_HANDS = 6

mp_hands = mp.solutions.hands #Detecting hands and landmarks
mp_drawing = mp.solutions.drawing_utils #Drawing hand skeleton

count_history = deque(maxlen=5)


def parse_arguments():
    """
    Reads command-line arguments.

    Example:
    python finger_counter.py --max-hands 8
    """

    parser = argparse.ArgumentParser(
        description="Multi-hand rotation-aware finger counter using OpenCV and MediaPipe."
    )

    parser.add_argument(
        "--max-hands",
        type=int,
        default=DEFAULT_MAX_HANDS,
        help="Maximum number of hands MediaPipe should try to detect."
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index. Usually 0 for default webcam."
    )

    args = parser.parse_args()

    if args.max_hands < 1:
        args.max_hands = 1

    return args


def calculate_distance(point_a, point_b):
    """
    Calculates 2D distance between two MediaPipe landmarks.
    """

    dx = point_a.x - point_b.x
    dy = point_a.y - point_b.y

    return math.sqrt(dx ** 2 + dy ** 2)


def calculate_angle_3d(point_a, point_b, point_c):
    """
    Calculates the 3D angle ABC in degrees.
    point_b is the middle joint.
    """

    vector_ab = (
        point_a.x - point_b.x,
        point_a.y - point_b.y,
        point_a.z - point_b.z
    )

    vector_cb = (
        point_c.x - point_b.x,
        point_c.y - point_b.y,
        point_c.z - point_b.z
    )

    dot_product = (
        vector_ab[0] * vector_cb[0] +
        vector_ab[1] * vector_cb[1] +
        vector_ab[2] * vector_cb[2]
    )

    magnitude_ab = math.sqrt(
        vector_ab[0] ** 2 +
        vector_ab[1] ** 2 +
        vector_ab[2] ** 2
    )

    magnitude_cb = math.sqrt(
        vector_cb[0] ** 2 +
        vector_cb[1] ** 2 +
        vector_cb[2] ** 2
    )

    if magnitude_ab == 0 or magnitude_cb == 0:
        return 0

    cosine_angle = dot_product / (magnitude_ab * magnitude_cb)
    cosine_angle = max(-1.0, min(1.0, cosine_angle))

    return math.degrees(math.acos(cosine_angle))


def get_bounding_box(hand_landmarks):
    """
    Returns the normalized bounding box around a hand:
    min_x, min_y, max_x, max_y.
    """

    landmarks = hand_landmarks.landmark

    x_values = [landmark.x for landmark in landmarks]
    y_values = [landmark.y for landmark in landmarks]

    return min(x_values), min(y_values), max(x_values), max(y_values)


def calculate_box_area(box):
    """
    Calculates bounding box area.
    Coordinates are normalized between 0 and 1.
    """

    min_x, min_y, max_x, max_y = box

    width = max(0, max_x - min_x)
    height = max(0, max_y - min_y)

    return width * height

#IoU: Intersection over Union, measures how much two boxes overlap.
def calculate_iou(box_a, box_b):
    """
    Calculates Intersection over Union between two bounding boxes.
    Used to detect duplicate hand detections.
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)

    intersection_area = inter_width * inter_height

    area_a = calculate_box_area(box_a)
    area_b = calculate_box_area(box_b)

    union_area = area_a + area_b - intersection_area

    if union_area == 0:
        return 0

    return intersection_area / union_area


def get_box_center(box):
    """
    Returns the center point of a bounding box.
    """

    min_x, min_y, max_x, max_y = box

    return (
        (min_x + max_x) / 2,
        (min_y + max_y) / 2
    )


def calculate_center_distance(box_a, box_b):
    """
    Calculates distance between two bounding box centers.
    """

    center_a = get_box_center(box_a)
    center_b = get_box_center(box_b)

    dx = center_a[0] - center_b[0]
    dy = center_a[1] - center_b[1]

    return math.sqrt(dx ** 2 + dy ** 2)


def get_visible_landmark_ratio(hand_landmarks):
    """
    Calculates the percentage of hand landmarks inside the camera frame.
    """

    landmarks = hand_landmarks.landmark
    visible_count = 0

    for landmark in landmarks:
        if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
            visible_count += 1

    return visible_count / len(landmarks)


def estimate_palm_size(hand_landmarks):
    """
    Estimates palm size using wrist and palm landmarks.
    This gives a scale reference for validation.
    """

    landmarks = hand_landmarks.landmark

    wrist = landmarks[0]
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    ring_mcp = landmarks[13]
    pinky_mcp = landmarks[17]

    palm_distances = [
        calculate_distance(wrist, index_mcp),
        calculate_distance(wrist, middle_mcp),
        calculate_distance(wrist, ring_mcp),
        calculate_distance(wrist, pinky_mcp),
        calculate_distance(index_mcp, pinky_mcp),
    ]

    return sum(palm_distances) / len(palm_distances)

#If the hand is for example half out of screen or cannot be identified, it is being warned.
def validate_hand_structure(hand_landmarks):
    """
    Checks whether a detected hand looks structurally valid.

    Returns:
    is_valid: bool
    status_message: str
    """

    landmarks = hand_landmarks.landmark

    visible_ratio = get_visible_landmark_ratio(hand_landmarks)

    if visible_ratio < 0.70:
        return False, "Partial hand"

    box = get_bounding_box(hand_landmarks)
    box_area = calculate_box_area(box)

    if box_area < 0.01:
        return False, "Too small"

    palm_size = estimate_palm_size(hand_landmarks)

    if palm_size < 0.03:
        return False, "Palm too small"

    index_to_pinky_width = calculate_distance(landmarks[5], landmarks[17])

    if index_to_pinky_width < palm_size * 0.25:
        return False, "Unclear palm structure"

    finger_chains = [
        [1, 2, 3, 4],       # Thumb
        [5, 6, 7, 8],       # Index
        [9, 10, 11, 12],    # Middle
        [13, 14, 15, 16],   # Ring
        [17, 18, 19, 20],   # Pinky
    ]

    valid_finger_chains = 0

    for chain in finger_chains:
        chain_length = 0

        for i in range(len(chain) - 1):
            chain_length += calculate_distance(
                landmarks[chain[i]],
                landmarks[chain[i + 1]]
            )

        if chain_length > palm_size * 0.25:
            valid_finger_chains += 1

    if valid_finger_chains < 4:
        return False, "Weak finger structure"

    return True, "Valid hand"


def filter_valid_unique_hands(hand_landmarks_list):
    """
    Filters detected hands by:
    1. structural validity
    2. duplicate overlap
    3. center distance

    Returns:
    valid_unique_hands: list
    rejected_reasons: list
    """

    valid_unique_hands = []
    valid_boxes = []
    rejected_reasons = []

    for hand_landmarks in hand_landmarks_list:
        is_valid, reason = validate_hand_structure(hand_landmarks)

        if not is_valid:
            rejected_reasons.append(reason)
            continue

        current_box = get_bounding_box(hand_landmarks)
        is_duplicate = False

        for existing_box in valid_boxes:
            iou = calculate_iou(current_box, existing_box)
            center_distance = calculate_center_distance(current_box, existing_box)

            if iou > 0.60 or (iou > 0.25 and center_distance < 0.08):
                is_duplicate = True
                rejected_reasons.append("Duplicate hand")
                break

        if not is_duplicate:
            valid_unique_hands.append(hand_landmarks)
            valid_boxes.append(current_box)

    return valid_unique_hands, rejected_reasons


def count_open_fingers(hand_landmarks):
    """
    Counts open fingers using 3D joint angles.

    This avoids relying on MediaPipe's Right/Left hand label,
    which can flip when the hand rotates.
    """

    landmarks = hand_landmarks.landmark
    fingers = []

    thumb_angle = calculate_angle_3d(
        landmarks[2],
        landmarks[3],
        landmarks[4]
    )

    thumb_is_open = thumb_angle > 145
    fingers.append(1 if thumb_is_open else 0)

    finger_joint_sets = [
        (5, 6, 8),      # Index
        (9, 10, 12),    # Middle
        (13, 14, 16),   # Ring
        (17, 18, 20),   # Pinky
    ]

    for mcp_id, pip_id, tip_id in finger_joint_sets:
        finger_angle = calculate_angle_3d(
            landmarks[mcp_id],
            landmarks[pip_id],
            landmarks[tip_id]
        )

        finger_is_open = finger_angle > 155
        fingers.append(1 if finger_is_open else 0)

    return sum(fingers), fingers


def calculate_hand_rotation(hand_landmarks):
    """
    Estimates hand rotation using the wrist and middle finger MCP landmark.
    """

    landmarks = hand_landmarks.landmark

    wrist = landmarks[0]
    middle_mcp = landmarks[9]

    dx = middle_mcp.x - wrist.x
    dy = middle_mcp.y - wrist.y

    rotation_angle = math.degrees(math.atan2(dx, -dy))

    return rotation_angle


def describe_rotation(rotation_angle):
    """
    Converts rotation angle into a readable label.
    """

    if rotation_angle > 60:
        return "Sideways Right"
    elif rotation_angle > 25:
        return "Tilted Right"
    elif rotation_angle < -60:
        return "Sideways Left"
    elif rotation_angle < -25:
        return "Tilted Left"
    else:
        return "Upright"


def get_stable_count(current_count):
    """
    Reduces flickering by using the most common count
    from the last few frames.
    """

    count_history.append(current_count)

    return max(set(count_history), key=count_history.count)


def draw_interface(frame, stable_total, raw_total, hand_results, rejected_reasons, max_hands):
    """
    Draws all text information on the video frame.
    """

    cv2.putText(
        frame,
        f"Detected Open Fingers: {stable_total}",
        (10, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        f"Raw Count: {raw_total}",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Hands Detected: {len(hand_results)} / {max_hands}",
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    if rejected_reasons:
        unique_reasons = sorted(set(rejected_reasons))
        rejected_text = "Rejected: " + ", ".join(unique_reasons)

        cv2.putText(
            frame,
            rejected_text,
            (10, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2
        )

        y_position = 175
    else:
        y_position = 145

    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    for hand_result in hand_results:
        cv2.putText(
            frame,
            (
                f"{hand_result['name']}: "
                f"{hand_result['count']} open | "
                f"{hand_result['rotation_label']} "
                f"({hand_result['rotation_angle']:.1f} deg)"
            ),
            (10, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        y_position += 25

        finger_status_text = []

        for finger_name, state in zip(finger_names, hand_result["finger_states"]):
            status = "Open" if state == 1 else "Closed"
            finger_status_text.append(f"{finger_name}:{status}")

        cv2.putText(
            frame,
            " | ".join(finger_status_text),
            (10, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1
        )

        y_position += 35


def main(max_hands, camera_index):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        print("Try running with a different camera index, for example: --camera 1")
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_hands,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while True:
            success, frame = cap.read()

            if not success:
                print("Error: Could not read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            total_open_fingers = 0
            hand_results = []
            rejected_reasons = []

            if results.multi_hand_landmarks:
                valid_hands, rejected_reasons = filter_valid_unique_hands(
                    results.multi_hand_landmarks
                )

                for hand_index, hand_landmarks in enumerate(valid_hands):
                    hand_count, finger_states = count_open_fingers(hand_landmarks)

                    rotation_angle = calculate_hand_rotation(hand_landmarks)
                    rotation_label = describe_rotation(rotation_angle)

                    total_open_fingers += hand_count

                    hand_results.append(
                        {
                            "name": f"Hand {hand_index + 1}",
                            "count": hand_count,
                            "finger_states": finger_states,
                            "rotation_angle": rotation_angle,
                            "rotation_label": rotation_label
                        }
                    )

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
            else:
                count_history.clear()

            stable_total = get_stable_count(total_open_fingers)

            draw_interface(
                frame=frame,
                stable_total=stable_total,
                raw_total=total_open_fingers,
                hand_results=hand_results,
                rejected_reasons=rejected_reasons,
                max_hands=max_hands
            )

            cv2.imshow("Dynamic Multi-Hand Finger Counter", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments.max_hands, arguments.camera)