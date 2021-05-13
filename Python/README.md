# Python implementation

PyTorch implementation of MarkerPose. Training of the SuperPoint-like network and EllipSegNet can be done with `train_superpoint.py` and `train_ellipsegnet.py`, respectively. The pose estimation example with a robotic arm is implemented in `robot_test.py`.

## Dependencies

* Python 3
* NumPy
* OpenCV
* PyTorch
* Pandas

## Usage

Clone this repo and download the dataset example as explained [here](https://github.com/jhacsonmeza/MarkerPose#pose-estimation-example). The folder structure is important to execute successfully the example. To run the pose estimation example with the robotic arm:

```
cd Python
python robot_test.py
```