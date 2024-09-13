import cv2
import numpy as np

LINES_HAND = [[0,1],[1,2],[2,3],[3,4], 
            [0,5],[5,6],[6,7],[7,8],
            [5,9],[9,10],[10,11],[11,12],
            [9,13],[13,14],[14,15],[15,16],
            [13,17],[17,18],[18,19],[19,20],[0,17]]

# LINES_BODY to draw the body skeleton when Body Pre Focusing is used
LINES_BODY = [[4,2],[2,0],[0,1],[1,3],
            [10,8],[8,6],[6,5],[5,7],[7,9],
            [6,12],[12,11],[11,5],
            [12,14],[14,16],[11,13],[13,15]]

class HandTrackerRenderer:
    def __init__(self, 
                tracker,
                output=None):

        self.tracker = tracker

        # Rendering flags
        self.show_pd_box = False
        self.show_pd_kps = False
        self.show_rot_rect = False
        self.show_handedness = 0
        self.show_landmarks = True
        self.show_scores = False
     

        self.show_xyz_zone = self.show_xyz = True
        self.show_body = False # self.tracker.body_pre_focusing is not None
        self.show_inferences_status = False

        if output is None:
            self.output = None
        else:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.output = cv2.VideoWriter(output,fourcc,self.tracker.video_fps,(self.tracker.img_w, self.tracker.img_h)) 

    def norm2abs(self, x_y):
        x = int(x_y[0] * self.tracker.frame_size - self.tracker.pad_w)
        y = int(x_y[1] * self.tracker.frame_size - self.tracker.pad_h)
        return (x, y)

    def draw_hand(self, hand):

        # (info_ref_x, info_ref_y): coords in the image of a reference point 
        # relatively to which hands information (score, handedness, xyz,...) are drawn
        info_ref_x = hand.landmarks[0,0]
        info_ref_y = np.max(hand.landmarks[:,1])

        # thick_coef is used to adapt the size of the draw landmarks features according to the size of the hand.
        thick_coef = hand.rect_w_a / 400
        if hand.lm_score > self.tracker.lm_score_thresh:
            if self.show_rot_rect:
                cv2.polylines(self.frame, [np.array(hand.rect_points)], True, (0,255,255), 2, cv2.LINE_AA)
            if self.show_landmarks:
                lines = [np.array([hand.landmarks[point] for point in line]).astype(np.int32) for line in LINES_HAND]
                if self.show_handedness == 3:
                    color = (0,255,0) if hand.handedness > 0.5 else (0,0,255)
                else:
                    color = (255, 0, 0)
                cv2.polylines(self.frame, lines, False, color, int(1+thick_coef*3), cv2.LINE_AA)
                radius = int(1+thick_coef*5)
                    
                if self.show_handedness == 2:
                    color = (0,255,0) if hand.handedness > 0.5 else (0,0,255)
                elif self.show_handedness == 3:
                    color = (255, 0, 0)
                else: 
                    color = (0,128,255)
                for x,y in hand.landmarks[:,:2]:
                    cv2.circle(self.frame, (int(x), int(y)), radius, color, -1)

            if self.show_handedness == 1:
                cv2.putText(self.frame, f"{hand.label.upper()} {hand.handedness:.2f}", 
                        (info_ref_x-90, info_ref_y+40), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0) if hand.handedness > 0.5 else (0,0,255), 2)
            if self.show_scores:
                cv2.putText(self.frame, f"Landmark score: {hand.lm_score:.2f}", 
                        (info_ref_x-90, info_ref_y+110), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)

        if hand.pd_box is not None:
            box = hand.pd_box
            box_tl = self.norm2abs((box[0], box[1]))
            box_br = self.norm2abs((box[0]+box[2], box[1]+box[3]))
            if self.show_pd_box:
                cv2.rectangle(self.frame, box_tl, box_br, (0,255,0), 2)
            if self.show_pd_kps:
                for i,kp in enumerate(hand.pd_kps):
                    x_y = self.norm2abs(kp)
                    cv2.circle(self.frame, x_y, 6, (0,0,255), -1)
                    cv2.putText(self.frame, str(i), (x_y[0], x_y[1]+12), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
            if self.show_scores:
                x, y = info_ref_x - 90, info_ref_y + 80
                cv2.putText(self.frame, f"Palm score: {hand.pd_score:.2f}", 
                        (x, y), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
            
    def draw_body(self, body):
        lines = [np.array([body.keypoints[point] for point in line]) for line in LINES_BODY if body.scores[line[0]] > self.tracker.body_score_thresh and body.scores[line[1]] > self.tracker.body_score_thresh]
        cv2.polylines(self.frame, lines, False, (255, 144, 30), 2, cv2.LINE_AA)

    def draw(self, frame, hands):
        self.frame = frame
        for hand in hands:
            self.draw_hand(hand)
        return self.frame

    def exit(self):
        if self.output:
            self.output.release()
        cv2.destroyAllWindows()

    def waitKey(self, delay=1):
        cv2.imshow("Hand tracking", self.frame)
        if self.output:
            self.output.write(self.frame)
        key = cv2.waitKey(delay) 
        if key == 32:
            # Pause on space bar
            key = cv2.waitKey(0)
            if key == ord('s'):
                print("Snapshot saved in snapshot.jpg")
                cv2.imwrite("snapshot.jpg", self.frame)
        elif key == ord('1'):
            self.show_pd_box = not self.show_pd_box
        elif key == ord('2'):
            self.show_pd_kps = not self.show_pd_kps
        elif key == ord('3'):
            self.show_rot_rect = not self.show_rot_rect
        elif key == ord('4') and self.tracker.use_lm:
            self.show_landmarks = not self.show_landmarks
        elif key == ord('5') and self.tracker.use_lm:
            self.show_handedness = (self.show_handedness + 1) % 4
        elif key == ord('6'):
            self.show_scores = not self.show_scores
        elif key == ord('7') and self.tracker.use_lm:
            if self.tracker.use_gesture:
                self.show_gesture = not self.show_gesture
        elif key == ord('8'):
            if self.tracker.xyz:
                self.show_xyz = not self.show_xyz    
        elif key == ord('9'):
            if self.tracker.xyz:
                self.show_xyz_zone = not self.show_xyz_zone 
        elif key == ord('f'):
            self.show_fps = not self.show_fps
        elif key == ord('b'):
            try:
                if self.tracker.body_pre_focusing:
                    self.show_body = not self.show_body 
            except:
                pass
        elif key == ord('s'):
            self.show_inferences_status = not self.show_inferences_status
        return key
