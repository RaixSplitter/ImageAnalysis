import time
import cv2
import numpy as np
from skimage.util import img_as_float
from skimage.util import img_as_ubyte


def show_in_moved_window(win_name, img, x, y):
    """
    Show an image in a window, where the position of the window can be given
    """
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, x, y)
    cv2.imshow(win_name,img)


def capture_from_camera_and_show_images():
    
    ALPHA = 0.95
    THRESHOLD = 0.1
    ALERT_THRESHOLD = 0.05
    ALERT = False
    
    print("Starting image capture")

    print("Opening connection to camera")
    url = 0
    use_droid_cam = False
    if use_droid_cam:
        url = "http://192.168.1.120:4747/video"
    cap = cv2.VideoCapture(url)
    # cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Starting camera loop")
    # Get first image
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame")
        exit()

    # Transform image to gray scale and then to float, so we can do some processing
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = img_as_float(frame_gray)
    
    # Init background image
    bg_img = frame_gray
    
    # To keep track of frames per second
    start_time = time.time()
    n_frames = 0
    stop = False
    while not stop:
        ret, new_frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        # Transform image to gray scale and then to float, so we can do some processing
        new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        new_frame_gray = img_as_float(new_frame_gray)

        # Compute difference image
        dif_img = np.abs(new_frame_gray - frame_gray)
        
        # bg image
        bg_img = ALPHA * bg_img + (1 - ALPHA) * new_frame_gray
        
        # Compute difference image
        dif_bg_img = np.abs(new_frame_gray - bg_img)
        
        # Binary Threshhold
        # If pixel value do not meet threshold, set to 0 (black) else set to 1 (white)
        b_img = np.zeros(dif_bg_img.shape)
        b_img[dif_bg_img > THRESHOLD] = 1
        
        # Count the number of pixels above threshold
        n_F = np.sum(b_img)
        
        # Compute the number of foreground pixels n_F divided by the total number of pixels
        p_F = n_F / (b_img.shape[0] * b_img.shape[1])
        
        # If p_F is above ALERT_THRESHOLD, then we have a change
        ALERT = p_F > ALERT_THRESHOLD
        
        
        

        # Keep track of frames-per-second (FPS)
        n_frames = n_frames + 1
        elapsed_time = time.time() - start_time
        fps = int(n_frames / elapsed_time)

        # Put the FPS on the new_frame
        str_out = f"fps: {fps}"
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(new_frame, str_out, (50, 50), font, 1, 255, 1)
        
        # If ALERT is true, then we have a change
        
        if ALERT:
            str_out = f"Change Detected!"
            cv2.putText(new_frame, str_out, (50, 100), font, 1, (0, 0, 255), 1)
        else:
            str_out = f"No change"
            cv2.putText(new_frame, str_out, (50, 100), font, 1, (0, 255, 0), 1)
            
        # Stats
        mu = dif_img.mean()
        sigma = dif_img.std()
        min_val = dif_img.min()
        max_val = dif_img.max()
        
        str_out1 = f"mu: {mu:.2f} sigma: {sigma:.2f}"
        str_out2 = f"min: {min_val:.2f} max: {max_val:.2f}"
        cv2.putText(new_frame, str_out1, (50, 150), font, 1, 255, 1)
        cv2.putText(new_frame, str_out2, (50, 200), font, 1, 255, 1)
        
        # Display the resulting frame
        show_in_moved_window('Input', new_frame, 0, 10)
        show_in_moved_window('Input gray', new_frame_gray, 600, 10)
        show_in_moved_window('Difference image', dif_img, 1200, 10)
        show_in_moved_window('Background image', bg_img, 0, 510)
        show_in_moved_window('Difference bg image', dif_bg_img, 600, 510)
        show_in_moved_window('Binary Image', b_img, 1200, 510)

        # Old frame is updated
        frame_gray = new_frame_gray

        if cv2.waitKey(1) == ord('q'):
            stop = True

    print("Stopping image loop")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture_from_camera_and_show_images()
