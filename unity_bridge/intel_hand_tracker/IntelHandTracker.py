import array
from time import sleep
import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import mediapipe_utils as mpu

base_id = 12357


class IntelHandTracker:
    def __init__(
        self, device_id=base_id, internal_frame_height=480, internal_frame_width=640
    ):
        self.serial_number = None
        context = rs.context()
        devices = context.query_devices()
        if len(devices) == 0:
            sleep(1)
            print("\nNo devices connected. Exiting.")
            self.created = False
            return None

        current_id = device_id - base_id
        if len(devices) > current_id:
            self.serial_number = devices[current_id].get_info(
                rs.camera_info.serial_number
            )

        if self.serial_number == None:
            print("\nAll devices in use. Exiting.")
            self.created = False
            return None

        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial_number)

        self.internal_frame_width = internal_frame_width
        self.internal_frame_height = internal_frame_height

        # Configure the pipeline to stream color and depth frames
        config.enable_stream(
            rs.stream.depth,
            internal_frame_width,
            internal_frame_height,
            rs.format.z16,
            30,
        )
        config.enable_stream(
            rs.stream.color,
            internal_frame_width,
            internal_frame_height,
            rs.format.bgr8,
            30,
        )
        # Start the pipeline
        try:
            self.pipeline.start(config)
        except rs.error as e:
            self.created = False
            return None

        # Initialize MediaPipe Hands
        self.handsTracker = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
        )
        sleep(2)
        print("\nConnected to a device with serial number " + self.serial_number + ".")
        self.created = True

    def next_frame(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            return

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Convert the BGR image to RGB for MediaPipe
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Process the image and find hands
        results = self.handsTracker.process(rgb_image)

        i = 0
        hands = []
        if results.multi_hand_landmarks:
            hands = [dict] * len(results.multi_hand_landmarks)
            for hand_landmarks in results.multi_hand_landmarks:
                hands[i] = mpu.HandRegion()
                j = 0
                landmark_array = [array] * 21
                for landmark_data in hand_landmarks.landmark:
                    # Calculate 3D coordinates for each landmark
                    landmark = landmark_data
                    # Convert normalized coordinates to image coordinates
                    pixel_x = int(landmark.x * color_frame.get_width())
                    pixel_y = int(landmark.y * color_frame.get_height())

                    # Get the depth value at the landmark's position
                    if (
                        pixel_x < 0
                        or pixel_x >= self.internal_frame_width
                        or pixel_y < 0
                        or pixel_y >= self.internal_frame_height
                    ):
                        hands = []
                        i = -1
                        break
                    # print (pixel_y)
                    depth = depth_frame.get_distance(pixel_x, pixel_y)

                    # Get depth frame intrinsics
                    depth_intrinsics = (
                        depth_frame.profile.as_video_stream_profile().intrinsics
                    )

                    # Convert from pixel coordinates to 3D coordinates
                    position = rs.rs2_deproject_pixel_to_point(
                        depth_intrinsics, [pixel_x, pixel_y], depth
                    )
                    if j == 0:
                        hands[i].xyz = position

                    landmark_array[j] = [float] * 3
                    landmark_array[j][0] = position[0]
                    landmark_array[j][1] = position[1]
                    landmark_array[j][2] = position[2]
                    j = j + 1
                if i >= 0:
                    hands[i].rotated_world_landmarks = landmark_array
                    hands[i].label = (
                        "left"
                        if results.multi_handedness[i].classification[0].label
                        == "Right"
                        else "right"
                    )
                    print(hands[i].label)
                    i = i + 1
                else:
                    break
        return color_image, hands, self.serial_number
        # return color_image, self.hands, self.device.getMxId()

    def exit(self):
        # Stop the pipeline
        self.pipeline.stop()
        cv2.destroyAllWindows()