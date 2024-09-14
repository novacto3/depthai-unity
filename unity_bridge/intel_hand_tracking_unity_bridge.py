#!/usr/bin/env python3

# Adjust python path for UnityBridge
import sys
import os
import cv2

# -- UB
# Get the absolute path of the unity_bridge directory
# __file__ is the path to the current script
script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
# Add the unity_bridge directory to sys.path
sys.path.append(script_dir)
sys.path.append(script_dir+'/intel_hand_tracker')
# -- UB

from HandTrackerRenderer import HandTrackerRenderer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--port', type=int, help="Port")
parser_tracker = parser.add_argument_group("Tracker arguments")
parser_tracker.add_argument('--internal_frame_height', type=int,                                                                                 
                    help="Internal color camera frame height in pixels")
parser_tracker.add_argument('--internal_frame_width', type=int,                                                                                 
                    help="Internal color camera frame width in pixels")  
args = parser.parse_args()
dargs = vars(args)
tracker_args = {a:dargs[a] for a in ['internal_frame_height', 'internal_frame_width'] if dargs[a] is not None}

from IntelHandTracker import IntelHandTracker


# -- UB
from unity_bridge import UnityBridge, TestObject
# Unity Bridge Configuration
# Example usage in the main application
address = ('127.0.0.1', args.port)
unity_bridge = UnityBridge(address)
unity_bridge.start()
test_object = TestObject(result="Success")
# -- UB

tracker = IntelHandTracker(args.port, **tracker_args)
if not tracker.created:
    sys.exit()

#renderer = HandTrackerRenderer(tracker=tracker)
count = 0

while True:
    # Run hand tracker on next frame
    # 'bag' contains some information related to the frame 
    # and not related to a particular hand like body keypoints in Body Pre Focusing mode
    # Currently 'bag' contains meaningful information only when Body Pre Focusing is used
    frame, hands, serialNumber = tracker.next_frame()
    if frame is None: break
        # Draw hands
    #frame = renderer.draw(frame, hands)

    # -- UB
    # Prepare data for serialization
    test_object.arr1 = [count]
    count = count+1
    if len(hands)==1:
        names = ['hand_0','res2']
        objects = [hands[0],test_object]
        configs = [['label','xyz','rotated_world_landmarks'],['result','arr1']]  # List of fields to serialize for each object
    elif len(hands)==2:
        names = ['hand_0','hand_1','res2']
        objects = [hands[0],hands[1],test_object]
        configs = [['label','xyz','rotated_world_landmarks'],['xyz','label','rotated_world_landmarks'],['result','arr1']]  # List of fields to serialize for each object
    else:
        names = ['res2']
        objects = [test_object]
        configs = [['result','arr1']]  # List of fields to serialize for each object

    # Send data back to Unity
    #frame_ub = cv2.resize(frame,(576,324))
    unity_bridge.send(names, objects, configs, serialNumber)
    # -- UB

    #key = renderer.waitKey(delay=1)
    #if key == 27 or key == ord('q'):
    #    break
#renderer.exit()
tracker.exit()
