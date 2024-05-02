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
sys.path.append(script_dir+'/depthai_hand_tracker')
# -- UB

from HandTrackerRenderer import HandTrackerRenderer
import argparse




parser = argparse.ArgumentParser()
parser.add_argument('-p', '--port', type=int, help="Port")
parser.add_argument('-e', '--edge', action="store_true",
                    help="Use Edge mode (postprocessing runs on the device)")
parser_tracker = parser.add_argument_group("Tracker arguments")
parser_tracker.add_argument('-i', '--input', type=str, 
                    help="Path to video or image file to use as input (if not specified, use OAK color camera)")
parser_tracker.add_argument("--pd_model", type=str,
                    help="Path to a blob file for palm detection model")
parser_tracker.add_argument('--no_lm', action="store_true", 
                    help="Only the palm detection model is run (no hand landmark model)")
parser_tracker.add_argument("--lm_model", type=str,
                    help="Landmark model 'full', 'lite', 'sparse' or path to a blob file")
parser_tracker.add_argument('--use_world_landmarks', action="store_true", 
                    help="Fetch landmark 3D coordinates in meter")
parser_tracker.add_argument('-s', '--solo', action="store_true", 
                    help="Solo mode: detect one hand max. If not used, detect 2 hands max (Duo mode)")                    
parser_tracker.add_argument('-xyz', "--xyz", action="store_true", 
                    help="Enable spatial location measure of palm centers")
parser_tracker.add_argument('-g', '--gesture', action="store_true", 
                    help="Enable gesture recognition")
parser_tracker.add_argument('-c', '--crop', action="store_true", 
                    help="Center crop frames to a square shape")
parser_tracker.add_argument('-f', '--internal_fps', type=int, 
                    help="Fps of internal color camera. Too high value lower NN fps (default= depends on the model)")                    
parser_tracker.add_argument("-r", "--resolution", choices=['full', 'ultra'], default='full',
                    help="Sensor resolution: 'full' (1920x1080) or 'ultra' (3840x2160) (default=%(default)s)")
parser_tracker.add_argument('--internal_frame_height', type=int,                                                                                 
                    help="Internal color camera frame height in pixels")   
parser_tracker.add_argument("-lh", "--use_last_handedness", action="store_true",
                    help="Use last inferred handedness. Otherwise use handedness average (more robust)")                            
parser_tracker.add_argument('--single_hand_tolerance_thresh', type=int, default=10,
                    help="(Duo mode only) Number of frames after only one hand is detected before calling palm detection (default=%(default)s)")
parser_tracker.add_argument('--dont_force_same_image', action="store_true",
                    help="(Edge Duo mode only) Don't force the use the same image when inferring the landmarks of the 2 hands (slower but skeleton less shifted)")
parser_tracker.add_argument('-lmt', '--lm_nb_threads', type=int, choices=[1,2], default=2, 
                    help="Number of the landmark model inference threads (default=%(default)i)")  
parser_tracker.add_argument('-t', '--trace', type=int, nargs="?", const=1, default=0, 
                    help="Print some debug infos. The type of info depends on the optional argument.")                
parser_renderer = parser.add_argument_group("Renderer arguments")
parser_renderer.add_argument('-o', '--output', 
                    help="Path to output video file")
args = parser.parse_args()
dargs = vars(args)
tracker_args = {a:dargs[a] for a in ['pd_model', 'lm_model', 'internal_fps', 'internal_frame_height'] if dargs[a] is not None}

if args.edge:
    from HandTrackerEdge import HandTracker
    tracker_args['use_same_image'] = not args.dont_force_same_image
else:
    from HandTracker import HandTracker


# -- UB
from unity_bridge import UnityBridge, TestObject
# Unity Bridge Configuration
# Example usage in the main application
address = ('127.0.0.1', args.port)
unity_bridge = UnityBridge(address)
unity_bridge.start()
test_object = TestObject(result="Success")
# -- UB

tracker = HandTracker(
        input_src=args.input, 
        use_lm= not args.no_lm, 
        use_world_landmarks=args.use_world_landmarks,
        use_gesture=args.gesture,
        xyz=args.xyz,
        solo=args.solo,
        crop=args.crop,
        resolution=args.resolution,
        stats=True,
        trace=args.trace,
        use_handedness_average=not args.use_last_handedness,
        single_hand_tolerance_thresh=args.single_hand_tolerance_thresh,
        lm_nb_threads=args.lm_nb_threads,
        **tracker_args
        )

#renderer = HandTrackerRenderer(
#        tracker=tracker,
#        output=args.output)


while True:
    # Run hand tracker on next frame
    # 'bag' contains some information related to the frame 
    # and not related to a particular hand like body keypoints in Body Pre Focusing mode
    # Currently 'bag' contains meaningful information only when Body Pre Focusing is used
    frame, hands, bag = tracker.next_frame()
    if frame is None: break
    # Draw hands
#    frame = renderer.draw(frame, hands, bag)

    # -- UB
    # Prepare data for serialization
    test_object.arr1 = [unity_bridge.count]
    #print(len(hands))
    if len(hands)==1:
        names = ['hand_0','res2']
        objects = [hands[0],test_object]
        configs = [['lm_score','label','xyz','get_rotated_world_landmarks'],['result','arr1']]  # List of fields to serialize for each object
    elif len(hands)==2:
        names = ['hand_0','hand_1','res2']
        objects = [hands[0],hands[1],test_object]
        configs = [['lm_score','label','xyz','get_rotated_world_landmarks'],['lm_score','xyz','label','get_rotated_world_landmarks'],['result','arr1']]  # List of fields to serialize for each object
    else:
        names = ['res2']
        objects = [test_object]
        configs = [['result','arr1']]  # List of fields to serialize for each object

    # Send data back to Unity
    #frame_ub = cv2.resize(frame,(576,324))
    unity_bridge.send(names, objects, configs)
    # -- UB

#    key = renderer.waitKey(delay=1)
  #  if key == 27 or key == ord('q'):
    #    break
#renderer.exit()
tracker.exit()
