import cv2
import torch
import numpy as np
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess
from yolox.data.datasets.coco_classes import COCO_CLASSES
from yolox.exp.build import get_exp

def main():
    try:
        # Load the model
        exp = get_exp('exps/default/yolox_nano_val.py', None)
        model = exp.get_model()
        checkpoint = torch.load("YOLOX_outputs/yolox_nano_val/best_ckpt.pth", map_location="cuda:0")
        model.load_state_dict(checkpoint["model"])
        model.eval()
        
        # Define the device
        if torch.cuda.is_available():
            device = 'cuda'
            print("CUDA SUCCESS")
        else:
            print("CUDA FAILED")
            return
        model.to(device)

        # Open the video capture
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open video stream")
            return

        # Transformation function
        val_transform = ValTransform()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Preprocess the frame
            img, _ = val_transform(frame, None, (640, 640))
            img = torch.from_numpy(img).unsqueeze(0).float().to(device)

            # Run the model
            with torch.no_grad():
                outputs = model(img)

            # Post-process the outputs
            outputs = postprocess(outputs, num_classes=80, conf_thre=0.001, nms_thre=0.65)

            # Draw bounding boxes
            if outputs[0] is not None:
                output = outputs[0].cpu().numpy()
                bboxes = output[:, 0:4]
                scores = output[:, 4]
                cls_ids = output[:, 6]

                for i, box in enumerate(bboxes):
                    if COCO_CLASSES[int(cls_ids[i])] == 'person' and scores[i] >= 0.3:
                        x0, y0, x1, y1 = box
                        cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
                        cv2.putText(frame, f'{COCO_CLASSES[int(cls_ids[i])]} {scores[i]:.2f}', 
                                    (int(x0), int(y0) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow('YOLOX Human Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        if cap and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

