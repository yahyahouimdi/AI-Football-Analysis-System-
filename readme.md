# Football Match Analyser

An experimental football video analysis project that uses Python and Ultralytics YOLO to detect football-related objects in match footage. The project compares a general-purpose pretrained detector with a model fine-tuned on a football-specific dataset, with a particular focus on improving ball detection.

The current implementation is a video object-detection pipeline. It reads a match video, runs YOLO inference frame by frame, and saves an annotated output video. Player tracking, tactical analysis, statistics extraction, and a complete application entry point are planned extensions rather than implemented features in the current repository.

## Project Motivation

The initial detector was able to identify people in match footage, but it detected the ball inconsistently. The project therefore uses a football-specific dataset to fine-tune a YOLO model for the following four classes:

| Class | Description |
| --- | --- |
| `ball` | The football. |
| `goalkeeper` | A goalkeeper. |
| `player` | An outfield player. |
| `referee` | A match official. |

The ball is considerably smaller than the other classes, so its detection is more sensitive to image quality, camera angle, occlusion, and annotation coverage.

## Detection Pipeline

```text
Match video
	|
	v
YOLO model
	|
	+--> Detected bounding boxes and class predictions
	|
	v
Annotated output video saved by Ultralytics
```

The repository includes two inference scripts:

- `yolo_infernce_before_training.py` loads the general `yolov8x` model.
- `yolo_infernce_after_training.py` loads the fine-tuned weights from `models/best.pt`.

Both scripts use `input_video/video.mp4` as their input and pass `save=True` to Ultralytics, which writes the prediction output under an automatically generated `runs/detect/predict*` directory.

## Before and After Comparison

The following images document the project workflow and the visual difference between the baseline and fine-tuned detectors.

### Original Input

The input is football match footage sourced from the DFL Bundesliga video dataset on Kaggle.

![Original football match video](assets/original_video.png)

Source: [DFL Bundesliga football videos on Kaggle](https://www.kaggle.com/datasets/saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv).

### Baseline YOLO Inference

The general-purpose YOLO model detects people reasonably well, but ball detections are rare and inconsistent in the match footage.

![Baseline YOLO detection result](assets/video_after_normal_yolo.png)

### Fine-Tuned YOLO Inference

The fine-tuned model is trained on football-specific annotations for players, goalkeepers, referees, and the ball. This image shows the output produced after applying the football-specific weights.

![Fine-tuned YOLO detection result](assets/video_after_finetuning.png)

The comparison is qualitative: the repository does not currently record a numeric baseline-versus-fine-tuned benchmark in this README.

## Repository Structure

```text
.
|-- main.py                              # Placeholder application entry point
|-- yolo_infernce_before_training.py     # Inference with pretrained YOLOv8x
|-- yolo_infernce_after_training.py      # Inference with fine-tuned weights
|-- requirement.txt                       # Python dependency list
|-- yolov8x.pt                            # Local YOLO weights, when provided
|-- input_video/
|   `-- video.mp4                         # Expected input video
|-- assets/
|   |-- original-video.png                # Input-video reference image
|   |-- video_after_normal_yolo.png       # Baseline output reference image
|   `-- video_after_finetuning.png        # Fine-tuned output reference image
|-- runs/
|   `-- detect/predict*/                  # Ultralytics prediction outputs
`-- training/
	|-- readme.md                         # Detailed training documentation
	|-- football_training_yolo_v5_local.ipynb
	|-- football_training_yolo_v5_colab.ipynb
	|-- assets/                            # Training metric charts
	`-- football-players-detection-1/     # Downloaded dataset and labels
```

## Requirements

- Python 3.8 or newer is recommended.
- A machine with enough CPU/GPU memory to run the selected YOLO model.
- CUDA-enabled PyTorch is recommended for practical training and faster inference, but is not required for a basic CPU run.
- The input video must exist at `input_video/video.mp4`, unless the scripts are edited to use another path.

Install the project dependency with:

```bash
pip install -r requirement.txt
```

The dependency file currently contains `ultralytics`, which installs the YOLO inference and training interface and its required dependencies.

## Running Inference

Run the baseline detector:

```bash
python yolo_infernce_before_training.py
```

Run the fine-tuned detector:

```bash
python yolo_infernce_after_training.py
```

Before running the fine-tuned script, place the trained weights at:

```text
models/best.pt
```

If the input video is missing, either script raises a `FileNotFoundError` and prints the expected path. Each script also prints the first prediction result and its detected bounding boxes to the terminal. The saved annotated video is placed in an Ultralytics prediction directory under `runs/detect/`.

## Training Workflow

The complete training workflow is documented in [`training/readme.md`](training/readme.md) and is available in both local and Google Colab notebooks.

### Dataset

The model is trained with version 1 of Roboflow's [Football Players Detection dataset](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc). The checked-in `data.yaml` defines four classes and the `train`, `valid`, and `test` image splits. The dataset is distributed under the `CC BY 4.0` license; follow the dataset page's attribution requirements when redistributing it.

### Training Configuration

The notebooks install `ultralytics` and `roboflow`, locate the nested dataset split directories, update the dataset paths, and train a `yolov5x` detector using:

```bash
yolo task=detect mode=train model=yolov5x.pt data=<dataset-location>/data.yaml epochs=100 imgsz=640
```

Training outputs are normally written under `runs/detect/`. The best checkpoint is usually found at `weights/best.pt`; copy it to `models/best.pt` for use by `yolo_infernce_after_training.py`.

For local training, use `training/football_training_yolo_v5_local.ipynb`. For machines without sufficient hardware, use `training/football_training_yolo_v5_colab.ipynb` in Google Colab. The local workflow reads the Roboflow key through an environment file; never commit a real API key.

## Training Evaluation

The charts below are stored in `training/assets/` and summarize the training and validation behavior of the fine-tuned detector.

### Loss and mAP Curves

| Training and Validation Losses | Mean Average Precision (mAP) |
| --- | --- |
| ![Training and validation loss curves](training/assets/training_losses.png) | ![Mean average precision curves](training/assets/map_curves.png) |

Loss curves show the error made during training and validation. A generally decreasing trend indicates that the model is learning useful representations. The mAP curves summarize localization and classification quality; higher values indicate more accurate detections.

### Precision, Recall, and Class Performance

| Precision and Recall | Per-Class Performance |
| --- | --- |
| ![Precision and recall curves](training/assets/precision_recall.png) | ![Per-class detection performance](training/assets/per_class_performance.png) |

Precision measures how many predicted detections are correct, while recall measures how many annotated objects are found. The per-class chart is particularly useful for checking whether the smaller and harder-to-detect ball performs differently from players, goalkeepers, and referees.

## Dataset Layout

The downloaded dataset is expected to follow this structure:

```text
football-players-detection-1/
|-- data.yaml
`-- football-players-detection-1/
	|-- train/
	|   |-- images/
	|   `-- labels/
	|-- valid/
	|   |-- images/
	|   `-- labels/
	`-- test/
		|-- images/
		`-- labels/
```

YOLO annotation files contain one object per line using normalized coordinates:

```text
class_id center_x center_y width height
```

## Troubleshooting

- **`Input video not found`:** confirm that the file is named `video.mp4` and is located in `input_video/`.
- **Fine-tuned weights cannot be loaded:** confirm that `models/best.pt` exists and is a valid Ultralytics checkpoint.
- **No prediction video appears:** inspect the terminal output for model or codec errors and check the generated `runs/detect/predict*` directory.
- **CUDA out-of-memory:** run inference on CPU, reduce the model or batch settings, or use Google Colab for training.
- **Training paths are duplicated:** rerun the dataset path-fixing section in the training notebook so `train`, `val`, and `test` point to the actual image directories.
- **The ball is still missed:** review the precision/recall and per-class charts, then consider more representative annotations, higher-resolution training, or additional football footage.

## Limitations and Next Steps

The current repository provides object detection and visual comparison only. It does not yet provide:

- persistent object tracking across frames;
- player identity assignment or team classification;
- ball trajectory, possession, or tactical statistics;
- a command-line interface with configurable input and output paths; or
- an implemented application workflow in `main.py`.

These are natural next steps if the project is extended from detection into a complete match-analysis system.

## Data Sources and Attribution

- Match footage: [DFL Bundesliga football videos](https://www.kaggle.com/datasets/saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv) on Kaggle.
- Training dataset: [Football Players Detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) on Roboflow, version 1, licensed under CC BY 4.0.