import os
from pathlib import Path
from ultralytics import YOLO


def train_segmentation():
    try:
        model = YOLO('../yolo11n-seg.pt')
        print("Starting YOLO11 Segmentation Training...")
        results = model.train(
            data='coco8-seg.yaml',
            epochs=3,
            imgsz=640,
            device='cpu',
            patience=3,
            batch=16,
            name='segmentation_model'
        )
        print("Segmentation training done!")
        print(f"Model saved at: {results.save_dir}")
        return results
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_segmentation():
    try:
        # Find the latest segmentation model automatically
        runs_dir = Path('../runs/segment')
        latest_model_dir = max(runs_dir.glob('segmentation_model*'), key=os.path.getctime)
        model_path = latest_model_dir / 'weights' / 'best.pt'

        model = YOLO(str(model_path))
        print("Running segmentation test...")
        results = model.predict(
            source='media/messy_table.jpg',
            save=True
        )
        print("Prediction completed!")
        return results
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    train_segmentation()
    test_segmentation()
