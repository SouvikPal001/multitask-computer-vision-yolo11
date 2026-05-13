from models.classification import train_classification, test_classification
from models.detection import train_detection, test_detection
from models.segmentation import train_segmentation, test_segmentation
from models.pose import train_pose, test_pose


def main():
    print("=" * 60)
    print("YOLO11 Multi-Task Computer Vision Pipeline")
    print("=" * 60)

    # Train all models
    print("\n[1/4] Training Classification Model...")
    train_classification()

    print("\n[2/4] Training Detection Model...")
    train_detection()

    print("\n[3/4] Training Segmentation Model...")
    train_segmentation()

    print("\n[4/4] Training Pose Estimation Model...")
    train_pose()

    print("\n" + "=" * 60)
    print("All models trained successfully!")
    print("=" * 60)

    # Test all models
    print("\n[Testing Phase]")
    print("\nTesting Classification Model...")
    test_classification()

    print("\nTesting Detection Model...")
    test_detection()

    print("\nTesting Segmentation Model...")
    test_segmentation()

    print("\nTesting Pose Estimation Model...")
    test_pose()

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
