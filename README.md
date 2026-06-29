# i2i Academy Computer Vision - Dynamic Multi-Hand Finger Counter

This project is a real-time Computer Vision application developed for the i2i Academy Computer Vision assignment. It uses a webcam to detect human hands, extract hand landmark coordinates, determine whether fingers are open or closed, and display the number of detected open fingers directly on the video window.

The application was built with Python, OpenCV, and MediaPipe.

---

## Assignment Objective

The original objective of this assignment was to build a real-time Computer Vision application using Python.

The required application should:

* Open the webcam and capture the live video feed frame by frame.
* Use a pre-trained Computer Vision library to detect a hand.
* Extract hand landmark or joint coordinates.
* Create logical conditions to determine whether each finger is open or closed.
* Display the final count of open fingers as live text directly on the video window.
* Provide clean, well-commented Python code.
* Include a short screen recording demonstrating that the application works.

---

## Extra Features Added

In addition to the original assignment requirements, this project includes several improvements to make the application more flexible, stable, and explainable.

### 1. Multi-Hand Support

The application can detect and count open fingers from multiple hands in the same frame, instead of being limited to only one hand.

### 2. Configurable Maximum Hand Count

The maximum number of hands can be changed from the terminal without editing the source code.

Example:

```bash
python finger_counter.py --max-hands 8
```

This makes the program more flexible for one person, multiple people, or larger demonstrations.

### 3. Configurable Camera Index

The camera source can also be changed from the terminal.

Example:

```bash
python finger_counter.py --camera 1
```

This is useful when using an external webcam.

### 4. 3D Angle-Based Finger Counting

The program counts open fingers using 3D joint angles instead of relying only on simple x/y coordinate comparisons.

This improves stability when the hand rotates, tilts, or moves sideways.

### 5. Hand Structure Validation

Before counting a detected hand, the program checks whether the detection looks structurally valid.

It checks:

* Landmark visibility
* Bounding box size
* Palm size
* Palm width
* Finger-chain lengths

This helps reject unreliable detections such as partial hands, very small hands, or weak landmark structures.

### 6. Duplicate Hand Filtering

Sometimes one real hand can be detected as two hands, especially when the hand is sideways.

To reduce this problem, the program compares detected hand bounding boxes using:

* Intersection over Union
* Bounding box center distance

Likely duplicate detections are removed before finger counting.

### 7. Hand Rotation Estimation

The program estimates the rotation of each detected hand using the wrist landmark and the middle finger MCP landmark.

The hand is classified as:

* Upright
* Tilted Left
* Tilted Right
* Sideways Left
* Sideways Right

### 8. Stable Count Display

The program stores recent finger counts and displays the most common value from recent frames.

This reduces flickering caused by frame-by-frame detection changes.

### 9. Rejected Detection Feedback

If a detected hand is rejected, the program displays the reason on the video window.

Possible rejection reasons include:

* Partial hand
* Too small
* Palm too small
* Unclear palm structure
* Weak finger structure
* Duplicate hand

This makes the application easier to debug and explain.

---

## Difference Between Required Work and Added Improvements

The required work focused on building a basic real-time finger counter using webcam input, hand detection, landmark extraction, finger-state logic, and live text display.

The added improvements focused on making the project more robust and unique. These improvements include multi-hand support, configurable settings, duplicate filtering, hand-structure validation, 3D angle-based finger counting, hand rotation estimation, count smoothing, and rejected detection feedback.

The final version therefore satisfies the original assignment requirements while also adding extra engineering features to improve stability, flexibility, and explainability.

---

## Technologies Used

* Python
* OpenCV
* MediaPipe
* argparse
* math
* collections.deque

---

## Project Structure

```text
i2i-Academy-ComputerVision-1/
│
├── finger_counter.py
├── requirements.txt
├── README.md
└── .gitignore
```

### File Descriptions

| File                | Description                      |
| ------------------- | -------------------------------- |
| `finger_counter.py` | Main Python application          |
| `requirements.txt`  | Python dependency list           |
| `README.md`         | Project documentation            |
| `.gitignore`        | Files and folders ignored by Git |

---

## How the Application Works

The application follows this pipeline:

```text
Webcam frame
→ OpenCV reads the frame
→ Frame is converted from BGR to RGB
→ MediaPipe detects hand landmarks
→ Custom validation checks whether each hand is reliable
→ Duplicate hand detections are removed
→ 3D joint angles are calculated for each finger
→ Open fingers are counted
→ The total count is smoothed using recent frames
→ Results are displayed on the video window
```

---

## Landmark Detection

MediaPipe Hands is used as the pre-trained Computer Vision model.

For every detected hand, MediaPipe provides 21 landmarks.

Important landmarks used in this project include:

| Landmark ID | Meaning           |
| ----------- | ----------------- |
| `0`         | Wrist             |
| `4`         | Thumb tip         |
| `8`         | Index finger tip  |
| `12`        | Middle finger tip |
| `16`        | Ring finger tip   |
| `20`        | Pinky finger tip  |
| `5`         | Index MCP         |
| `9`         | Middle MCP        |
| `13`        | Ring MCP          |
| `17`        | Pinky MCP         |

These landmarks are used to estimate hand structure, finger states, and hand rotation.

---

## Finger Counting Logic

The program uses 3D angle-based logic to determine whether fingers are open or closed.

For the thumb, the angle is calculated using landmarks:

```text
2 → 3 → 4
```

For the other fingers, the angle is calculated using:

```text
Index:  5 → 6  → 8
Middle: 9 → 10 → 12
Ring:   13 → 14 → 16
Pinky:  17 → 18 → 20
```

If the calculated angle is large enough, the finger is considered open.

This method is more rotation-tolerant than simple coordinate comparison.

---

## Hand Validation Logic

After MediaPipe detects candidate hands, the program validates each hand before counting fingers.

The validation process checks:

### 1. Landmark Visibility

The program checks how many landmarks are inside the camera frame.

If too many landmarks are outside the frame, the hand is rejected as a partial hand.

### 2. Bounding Box Area

A bounding box is created around the detected landmarks.

If the box is too small, the detection is considered unreliable.

### 3. Palm Size

Palm size is estimated using the wrist and palm landmarks.

This is used as a scale reference for other validation checks.

### 4. Palm Width

The distance between the index MCP and pinky MCP landmarks is checked.

If this distance is too small, the palm structure is considered unclear.

### 5. Finger Chain Lengths

Each finger is treated as a landmark chain.

If too few finger chains have reasonable length, the detection is rejected as a weak hand structure.

---

## Duplicate Hand Filtering

A real hand can sometimes be detected twice, especially when it is sideways or partially visible.

To reduce false counting, the program compares detected hand bounding boxes.

It uses:

* Intersection over Union
* Center distance between bounding boxes

If two detections overlap too much or have very close centers, one of them is treated as a duplicate and rejected.

---

## Hand Rotation Logic

The program estimates hand rotation using:

```text
Wrist landmark: 0
Middle MCP landmark: 9
```

The direction from the wrist to the middle MCP is used to calculate an approximate rotation angle.

The rotation is then classified as:

```text
Upright
Tilted Right
Sideways Right
Tilted Left
Sideways Left
```

This helps explain how the hand is positioned in the frame.

---

## Count Smoothing

Real-time detection may flicker between nearby values.

For example, the raw count may quickly change like this:

```text
5, 5, 4, 5, 5
```

To reduce flickering, the program stores the most recent counts and displays the most common value.

This produces a more stable result on the screen.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/i2i-Academy-ComputerVision-1.git
cd i2i-Academy-ComputerVision-1
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If PowerShell activation is blocked on Windows, you can run Python directly from the virtual environment:

```bash
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## Running the Application

### Default Run

```bash
python finger_counter.py
```

Or on Windows using the virtual environment directly:

```bash
.\venv\Scripts\python.exe finger_counter.py
```

### Run With a Custom Maximum Hand Count

```bash
python finger_counter.py --max-hands 8
```

Windows virtual environment version:

```bash
.\venv\Scripts\python.exe finger_counter.py --max-hands 8
```

### Run With a Different Camera

```bash
python finger_counter.py --camera 1
```

### Run With Both Options

```bash
python finger_counter.py --max-hands 8 --camera 1
```

Press `q` to close the webcam window.

---

## Example Output

The video window displays information similar to this:

```text
Detected Open Fingers: 5
Raw Count: 5
Hands Detected: 1 / 6
Hand 1: 5 open | Upright (2.4 deg)
Thumb:Open | Index:Open | Middle:Open | Ring:Open | Pinky:Open
```

If a detection is rejected, the program may display:

```text
Rejected: Partial hand
```

or:

```text
Rejected: Duplicate hand
```

---

## Command-Line Arguments

| Argument      | Description                                            | Default |
| ------------- | ------------------------------------------------------ | ------- |
| `--max-hands` | Maximum number of hands MediaPipe should try to detect | `6`     |
| `--camera`    | Camera index used by OpenCV                            | `0`     |

Examples:

```bash
python finger_counter.py --max-hands 10
```

```bash
python finger_counter.py --camera 1
```

```bash
python finger_counter.py --max-hands 10 --camera 1
```

---

## Limitations

This application depends on webcam quality and MediaPipe landmark predictions.

Accuracy may decrease when:

* Hands overlap heavily
* Hands move too fast
* Lighting is poor
* Hands are partially outside the frame
* Hands are very far from the camera
* Too many hands are detected at once
* Fingers are hidden or crossed

The application counts visually detected open fingers. It does not make anatomical or medical judgments.

The maximum number of hands is configurable, but it is not unlimited. Very high values may reduce performance and increase false detections.

---

## Notes About Accessibility and Health Conditions

The displayed count represents visually detected open fingers based on MediaPipe landmarks and custom angle logic.

It should not be interpreted as a biological or medical measurement. For example, if a person has fewer fingers due to a health condition, the program may still estimate standard hand landmarks because the underlying hand model expects a general hand structure.

For this reason, the application uses the phrase:

```text
Detected Open Fingers
```

instead of making claims about how many fingers a person actually has.

---

## Demo Requirement

A short screen recording should demonstrate:

* The program running from the terminal
* The webcam window opening
* Hand landmarks being drawn
* Finger count changing live
* At least one example of open and closed fingers

Optional extra demo:

* Multiple hands detected
* Hand rotation display
* Rejected duplicate or partial hand feedback

---

## GitHub Submission Notes

This repository should contain only source code and configuration files.

Do not upload the assignment instruction document, screenshots, or private classroom files to this repository.

Recommended repository name:

```text
i2i-Academy-ComputerVision-1
```

---

## Summary

This project satisfies the original Computer Vision assignment by detecting hands, extracting landmarks, determining open or closed fingers, and displaying the finger count live on the webcam feed.

It also adds extra features such as multi-hand support, configurable settings, hand-structure validation, duplicate filtering, rotation estimation, and count smoothing to make the application more stable and explainable.

---

## Author

Developed as part of the i2i Academy Computer Vision homework.
