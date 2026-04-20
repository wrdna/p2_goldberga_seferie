from __future__ import print_function
import sys
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist

def touches_border(cnt, w, h, margin=2):
    pts = cnt.reshape(-1, 2)
    xs = pts[:, 0]
    ys = pts[:, 1]

    return (
        np.any(xs <= margin) or
        np.any(xs >= w - 1 - margin) or
        np.any(ys <= margin) or
        np.any(ys >= h - 1 - margin)
    )

def detect_ball(depth_img):
    min_depth, max_depth = 200, 4000
    #  smoothing image. morphological close then open.
    depth_smooth = cv2.medianBlur(depth_img, 5)
    mask = cv2.inRange(depth_smooth, min_depth, max_depth)
    mask[depth_img == 0] = 0

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_idx = -1
    best_score = 1e9

    h, w = depth_img.shape[:2]

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        (x, y), r = cv2.minEnclosingCircle(cnt)

        bottom = by + bh
        expected_bottom = y + r
        if bottom > expected_bottom + 10:
            continue

        circle_area = np.pi * r * r
        fill_ratio = area / circle_area if circle_area > 0 else 0
        if fill_ratio < 0.5:
            continue

        if r < 3 or r > 200:
            continue

        perim = cv2.arcLength(cnt, True)
        if perim <= 0:
            continue

        pts = cnt.reshape(-1, 2).astype(np.float32)
        dists = np.sqrt((pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2)

        radial_error = np.mean(np.abs(dists - r))
        radial_std = np.std(dists)

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        hull_perim = cv2.arcLength(hull, True)

        if hull_area <= 700 or hull_perim <= 0:
            continue

        # circularity = 4 * np.pi * area / (perim * perim)
        hull_circularity = 4 * np.pi * hull_area / (hull_perim * hull_perim)
        solidity = area / hull_area

        # throwing away junk
        # on the border, we expect higher solidity.
        # not on the border, we can accept lower solidity
        border = touches_border(cnt, w, h)
        if not border:
            if hull_circularity < 0.6:
                continue

            if solidity < 0.6:
                continue
        else:
            if hull_circularity < 0.5:
                continue

            if solidity < 0.9:
                continue


        # lower is better
        score = (
            2.0 * radial_error
            + 2.0 * radial_std
            - 20.0 * hull_circularity
            - 20.0 * solidity
        )

        # we dont want the border to dominate
        if border:
            score += 0

        if score < best_score:
             best_score = score
             best_idx = i

    display = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    display_bgr = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

    best_contour = contours[best_idx] if best_idx != -1 else None

    ball_centroid = None
    ball_radius = None
    ball_depth = None

    if best_idx != -1:
        (x, y), r = cv2.minEnclosingCircle(best_contour)

        ball_centroid = (int(x), int(y))
        ball_radius = int(r)

        # depth of only the selected ball contour
        contour_mask = np.zeros(depth_img.shape[:2], dtype=np.uint8)
        cv2.drawContours(contour_mask, [best_contour], -1, 255, thickness=-1)
        vals = depth_img[(mask == 255) & (depth_img > 0)]

        if vals.size > 0:
            # raw average
            ball_depth = float(np.mean(vals))

            # better: trimmed average to reduce shadow/noise influence
            med = np.median(vals)
            tol = 300  # mm
            trimmed = vals[(vals > med - tol) & (vals < med + tol)]

            if trimmed.size > 0:
                ball_depth = float(np.mean(trimmed))

        cv2.circle(display_bgr, ball_centroid, ball_radius, (255, 0, 0), 2)
        cv2.circle(display_bgr, ball_centroid, 4, (0, 0, 255), -1)
        cv2.drawContours(display_bgr, [best_contour], -1, (0, 255, 0), 2)

    cv2.imshow("depth contours", display_bgr)
    cv2.waitKey(1)

    return ball_centroid, ball_radius, ball_depth


class ball_chase:
    def __init__(self):

        TOPIC_SUB_DEPTH = "/camera/depth/image_raw"
        TOPIC_PUB_DEPTH = "ball_depth_processed"
        TOPIC_PUB_CMD   = "/cmd_vel_mux/input/navi"

        self.image_pub = rospy.Publisher(TOPIC_PUB_DEPTH, Image, queue_size=1)
        self.image_sub = rospy.Subscriber(TOPIC_SUB_DEPTH, Image, self.callback)
        self.cmd_pub = rospy.Publisher(
            TOPIC_PUB_CMD,
            Twist,
            queue_size=1
        )

        self.bridge = CvBridge()
        self.filtered_depth = None
        self.prev_linear = 0.0
        self.moving_forward = False

    def callback(self, data):
        try:
            depth_img = self.bridge.imgmsg_to_cv2(data, '16UC1')
        except CvBridgeError as e:
            rospy.logerr(str(e))
            return

        ball_centroid, ball_radius, ball_depth = detect_ball(depth_img)

        cmd = Twist()

        h, w = depth_img.shape[:2]

        deadband = 0.01
        Kp_turn = 4
        max_turn = 3

        target_depth = 1000

        start_margin = 140
        stop_margin = 60

        Kp_forward = 0.08
        max_forward = 2
        max_reverse = -0.25

        alpha = 0.25
        max_delta_v = 0.08

        # ball lost: stop and reset state
        if ball_centroid is None or ball_depth is None:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

            self.filtered_depth = None
            self.prev_linear = 0.0
            self.moving_forward = False

            self.cmd_pub.publish(cmd)
            return

        ball_x, ball_y = ball_centroid

        image_center_x = w / 2.0
        error_x = ball_x - image_center_x
        error_norm = error_x / image_center_x

        # turn toward center
        if abs(error_norm) > deadband:
            cmd.angular.z = float(np.clip(
                -Kp_turn * error_norm,
                -max_turn,
                max_turn
            ))
        else:
            cmd.angular.z = 0.0

        if self.filtered_depth is None:
            self.filtered_depth = float(ball_depth)
        else:
            self.filtered_depth = (
                alpha * float(ball_depth)
                + (1.0 - alpha) * self.filtered_depth
            )

        depth = self.filtered_depth
        depth_error = depth - target_depth

        # hysteresis for forward movement
        if not self.moving_forward:
            if depth_error > start_margin:
                self.moving_forward = True
        else:
            if depth_error < stop_margin:
                self.moving_forward = False

        desired_linear = 0.0

        # forward only if mostly centered
        if self.moving_forward and abs(error_norm) < 0.25:
            desired_linear = float(np.clip(
                Kp_forward * depth_error,
                max_reverse,
                max_forward
            ))
        else:
            desired_linear = 0.0

        # ratelimit forward velocity
        delta_v = desired_linear - self.prev_linear
        delta_v = float(np.clip(delta_v, -max_delta_v, max_delta_v))

        cmd.linear.x = self.prev_linear + delta_v
        self.prev_linear = cmd.linear.x

        print(
            "x:", ball_x,
            "err:", round(error_norm, 3),
            "depth:", round(depth, 1),
            "raw_depth:", round(float(ball_depth), 1),
            "lin:", round(cmd.linear.x, 3),
            "ang:", round(cmd.angular.z, 3),
            "moving:", self.moving_forward
        )

        self.cmd_pub.publish(cmd)

def main(args):
    rospy.init_node('ball_chase', anonymous=True)
    bc = ball_chase()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)
