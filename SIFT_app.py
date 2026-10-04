#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import sys
import numpy as np


class My_App(QtWidgets.QMainWindow):

    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)

        self._cam_id = 0
        self._cam_fps = 10
        self._is_cam_enabled = False
        self._is_template_loaded = False

        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        self._camera_device = cv2.VideoCapture(self._cam_id)
        self._camera_device.set(3, 320)
        self._camera_device.set(4, 240)

        # Timer used to trigger the camera
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(int(1000 / self._cam_fps))

    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]
            pixmap = QtGui.QPixmap(self.template_path)
            self.template_label.setPixmap(pixmap)
            print("Loaded template image file: " + self.template_path)


            # Initial SIFT setup here:
            
            # Collects the template_path image as grayscale.
            self.img = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE) 

            sift = cv2.SIFT_create()

            ## @brief locates visual key points 
            #
            # @param self.img grayscaled image.
            # @param None no specific region to search
            # @return self.kp_template (Keypoints): A tuple of OpenCV KeyPoint objects 
            #           storing geometric metadata (pixel location, neighbourhood size, 
            #           and angles) about where landmarks were found in self.img
            # @return self.desc_template (Descriptors): A 2D NumPy array with shape (N, 128),
            #           where N is the number of keypoints detected.
            self.kp_template, self.desc_template = sift.detectAndCompute(self.img, None)

            ## @brief configures how FLANN indexes the template descriptors in memory
            #
            # @param algorithm=0: Selects FLANN_INDEX_KDTREE (0 is the enum value for KD-Tree in OpenCV). 
            #                      KD-Trees organize high-dimensional data (SIFT's 128-element 
            #                      descriptors) into binary search trees so nearest neighbors can be found 
            #                      without searching every item linearly.
            # @param trees=5: Sets the number of parallel randomized KD-trees to build. 5 is the standard
            #                 balance between index creation time and search accuracy for SIFT features.
            index_params = dict(algorithm=0, trees=5)
            search_params = dict()

            # Create matcher instance using index_params and (default) search_params as settings.
            self.flann = cv2.FlannBasedMatcher(index_params, search_params)
            self._is_template_loaded = True

    # Source: stackoverflow.com/questions/34232632/
    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(
            cv_img.data,
            width,
            height,
            bytesPerLine,
            QtGui.QImage.Format_RGB888,
        )
        return QtGui.QPixmap.fromImage(q_img)

    def SLOT_query_camera(self):
        sift = cv2.SIFT_create()

        ret, frame = self._camera_device.read()
        # TODO run SIFT on the captured frame

        if ret and self._is_template_loaded:
            pixmap = self.convert_cv_to_pixmap(frame)
            self.live_image_label.setPixmap(pixmap)

            grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Locates visual key points of live camera grayframe.
            kp_cap, desc_cap = sift.detectAndCompute(grayframe, None)

            ## @brief searches the live camera frame (desc_cap) for every landmark in the 
            #         template image (self.desc_template) and returns the top 2 most 
            #         similarly matched candidates for each template point.
            # @param self.desc_template template image
            # @param desc_cap live camera frame
            # @param k=2 find top 2 most similarly matched candidates
            # @return top 2 most similarly matched candidate points
            matches = self.flann.knnMatch(self.desc_template, desc_cap, k=2)

            good_points = []

            # Lowe's Ratio Test: 
            # If the #1 match (m) is much closer than the #2 match (n), it's a true, unique landmark match.
            # If the #1 match (m) and #2 match (n) are almost identical in distance, the feature is ambiguous
            # (like a blank wall or repeating pattern) and gets thrown out as noise.
            for m, n in matches:
                if m.distance < 0.6 * n.distance:
                    good_points.append(m)

            # Extract the (x, y) pixel coordinates of matched keypoints from the template image (queryIdx) 
            # converting them into the array shape OpenCV expects.
            query_pts = np.float32([self.kp_template[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)

            # Extract the (x, y) pixel coordinates of matched keypoints from the live camera image (trainIdx) 
            # converting them into the array shape OpenCV expects.
            train_pts = np.float32([kp_cap[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

            ## @brief Calculates the 3 x 3 transformation matrix that maps the template image onto the live frame. 
            #         RANSAC uses a 5-pixel threshold to reject false matches, generating mask 
            #         (an array of 1s for true inlier matches and 0s for false outlier matches).
            # @param query_pts pixel coordinates of matched keypoints from the template image
            # @param train_pts pixel coordinates of matched keypoints from the live camera image
            # @param cv2.RANSAC uses RANSAC algorithm to filter outliers
            # @param 5.0 defines the maximum allowable error (pixel) for a match to be classified as a valid inlier
            # @return matrix 3x3 transformed matrix
            # @return mask array of inliers and outliers
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)

            # Flattens that mask array into a 1D Python list.
            # Useful for drawing explicit feature-matching lines between the template and live frame using cv2.drawMatches().
            matches_mask = mask.ravel().tolist()

            # Perspective transform
            h, w = self.img.shape

            # Creates a list containing the 4 corners of the template image.
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

            # Uses original 4 corners and multiplies them by the Homography matrix
            # to project where those corners land on the live webcam frame.
            dst = cv2.perspectiveTransform(pts, matrix)

            # Draws a closed 4-sided bounding box connecting the transformed corners (dst) 
            # directly onto the camera frame in blue with a line thickness of 3 pixels.
            homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
            cv2.imshow("Homography", homography)


    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())