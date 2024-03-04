import cv2
import numpy as np
import tensorflow as tf

# Load TFLite model
model_path = 'model.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_tensor_index = interpreter.get_input_details()[0]['index']
output = interpreter.tensor(interpreter.get_output_details()[0]['index'])

# Open a connection to the webcam (camera index 0)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Preprocess the frame for the TFLite model
    input_shape = interpreter.get_input_details()[0]['shape'][1:3]
    input_data = cv2.resize(frame, tuple(input_shape))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = (input_data.astype(np.float32) - 127.0)/128.0   # Normalize to [-1, 1]

    # Set the input tensor
    interpreter.set_tensor(input_tensor_index, input_data)

    # Run inference
    interpreter.invoke()

    # Get the output
    output_data = output()[0]

    # Post-process the output if needed
    # ...

    # Display or use the result as needed
    # ...

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
