import cv2
import os

# === Configuration ===
video_path = 'recorded_video.mp4'   # Your video file
output_folder = 'dataset_frames_new'    # Folder to save images
frame_interval = 1                  # Save every 5th frame

# === Setup ===
os.makedirs(output_folder, exist_ok=True)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ Failed to open video: {video_path}")
    exit()

frame_id = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % frame_interval == 0:
        filename = os.path.join(output_folder, f'frame_{saved:04d}.jpg')
        cv2.imwrite(filename, frame)
        print(f"✅ Saved {filename}")
        saved += 1

    frame_id += 1

cap.release()
print(f"\n🎉 Done! Extracted {saved} frames to '{output_folder}'")
