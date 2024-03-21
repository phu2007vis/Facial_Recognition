import cv2
import requests
import numpy as np

# URL of the Flask server streaming the video
stream_url = "http://192.168.0.100:5000/image_feed"

# Function to receive frames from the Flask server
def receive_frames():
    # Send a GET request to the Flask server's video feed endpoint
    response = requests.get(stream_url, stream=True)
    if response.status_code == 200:
        # Read the raw bytes of the response
        bytes_data = bytes()
        for chunk in response.iter_content(chunk_size=1024):
            bytes_data += chunk
            # Check for frame boundary
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')
            if a != -1 and b != -1:
                # Extract frame and reset bytes_data
                frame_data = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                # Decode JPEG frame
                frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                yield frame

# Main function to display video stream
def main():
    for frame in receive_frames():
		
        cv2.imshow("Video Stream", frame)
        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
      

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
