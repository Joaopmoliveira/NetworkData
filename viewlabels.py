import cv2
import os
import glob

def view_yolo_dataset(dataset_path, split='train'):
    # Paths to images and labels based on standard YOLO layout
    img_dir = os.path.join(dataset_path, 'images', split)
    lbl_dir = os.path.join(dataset_path, 'labels', split)
    
    # Get list of all images
    image_files = glob.glob(os.path.join(img_dir, "*.png"))
    
    if not image_files:
        print(f"No images found in {img_dir}. Check your paths!")
        return

    print(f"Showing {len(image_files)} images. Press ANY KEY to next, ESC to exit.")

    for img_path in image_files:
        # 1. Load the image
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        # 2. Match image to its label file
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(lbl_dir, label_name)
        
        # 3. Read and draw labels if they exist
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    # YOLO format: class_id x_center y_center width height
                    data = line.strip().split()
                    if len(data) == 5:
                        _, x_norm, y_norm, _, _ = map(float, data)
                        
                        # Denormalize coordinates to pixel values
                        px = int(x_norm * w)
                        py = int(y_norm * h)
                        
                        # Draw a crosshair or circle on the point
                        cv2.drawMarker(img, (px, py), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
                        cv2.circle(img, (px, py), 3, (0, 255, 0), -1)

        # Display like your original on_print
        cv2.namedWindow("YOLO Dataset Check", cv2.WINDOW_NORMAL)
        cv2.imshow("YOLO Dataset Check", img)
        
        key = cv2.waitKey(0)
        if key == 27: # ESC key
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Point this to the 'yolo_dataset' folder created by your conversion script
    view_yolo_dataset("yolo_dataset", split='train')