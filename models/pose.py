import os
from pathlib import Path
from ultralytics import YOLO


def train_pose():
    try:
        model = YOLO('../yolo11n-pose.pt')
        print("Starting YOLO11 Pose Estimation Training...")
        results = model.train(
            data='coco8-pose.yaml',
            epochs=3,
            imgsz=640,
            device='cpu',
            patience=3,
            batch=16,
            name='pose_model'
        )
        print("Pose estimation training done!")
        print(f"Model saved at: {results.save_dir}")
        return results
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_pose():
    try:
        # Find the latest pose model automatically
        runs_dir = Path('../runs/pose')
        latest_model_dir = max(runs_dir.glob('pose_model*'), key=os.path.getctime)
        model_path = latest_model_dir / 'weights' / 'best.pt'

        model = YOLO(str(model_path))
        print("Running pose estimation test...")
        results = model.predict(
            source='media/poses.jpg',
            save=True
        )
        print("Prediction completed!")
        return results
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    train_pose()
    test_pose()
