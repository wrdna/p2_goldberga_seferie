#!/usr/bin/env python3

import math
import rospy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from tf.transformations import quaternion_from_euler

class GoalSequence:
    def __init__(self):
        rospy.init_node("goal_sequence")

        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
        rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.pose_callback)

        self.current_x = None
        self.current_y = None

        # L1 -> L2 -> L3 -> L1
        self.goals = [
            ("L1", 3.9130845069885254, 2.963785409927368, 0.0),
            ("L2", 2.119593620300293, -0.9486680030822754, 0.0),
            ("L3", 4.277632713317871, 0.9896430969238281, 0.0),
            ("L1", 3.9130845069885254, 2.963785409927368, 0.0),
        ]

        self.position_tolerance = 0.35
        self.goal_timeout = 120

    def pose_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

    def make_goal(self, x, y, yaw):
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"

        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0.0

        q = quaternion_from_euler(0, 0, yaw)
        goal.pose.orientation.x = q[0]
        goal.pose.orientation.y = q[1]
        goal.pose.orientation.z = q[2]
        goal.pose.orientation.w = q[3]

        return goal

    def distance_to_goal(self, x, y):
        if self.current_x is None or self.current_y is None:
            return None
        return math.sqrt((x - self.current_x) ** 2 + (y - self.current_y) ** 2)

    def wait_for_pose(self):
        rospy.loginfo("Waiting for /amcl_pose...")
        rate = rospy.Rate(5)
        while not rospy.is_shutdown() and self.current_x is None:
            rate.sleep()
        rospy.loginfo("Pose received.")

    def go_to_goal(self, label, x, y, yaw):
        goal = self.make_goal(x, y, yaw)
        rospy.loginfo(f"Sending {label}: x={x:.3f}, y={y:.3f}")
        self.goal_pub.publish(goal)

        start_time = rospy.Time.now()
        rate = rospy.Rate(5)

        while not rospy.is_shutdown():
            dist = self.distance_to_goal(x, y)

            if dist is not None:
                rospy.loginfo(f"{label} distance: {dist:.2f} m")
                if dist <= self.position_tolerance:
                    rospy.loginfo(f"{label} reached.")
                    return True

            elapsed = (rospy.Time.now() - start_time).to_sec()
            if elapsed > self.goal_timeout:
                rospy.logwarn(f"{label} timeout, moving to next goal.")
                return False

            rate.sleep()

    def run(self):
        self.wait_for_pose()
        rospy.sleep(2.0)

        for label, x, y, yaw in self.goals:
            self.go_to_goal(label, x, y, yaw)
            rospy.sleep(2.0)

        rospy.loginfo("Finished L1 -> L2 -> L3 -> L1")

if __name__ == "__main__":
    try:
        GoalSequence().run()
    except rospy.ROSInterruptException:
        pass
