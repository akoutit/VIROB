'''
Define image class which read the image, extract chessboard corners find the homography
'''
import cv2
import numpy as np


def normalize_trans(points):
    

    Args:
        points (np.ndarray): points to normalize, shape (n, 2)
    Return:
        np.ndarray: similarity transformation for normalizing these points, shape (3, 3)
    """
    pass


def homogenize(points):
    """Convert points to homogeneous coordinate

    Args:
        points (np.ndarray): shape (n, 2)
    Return:
        np.ndarray: points in homogeneous coordinate (with 1 padded), shape (n, 3)
    """
    re = np.ones((points.shape[0], 3))  # shape (n, 3)
    re[:, :2] = points
    return re


class Image:
    """Provide operations on image necessary for calibration"""
    refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def __init__(self, impath, square_size=0.03, debug=False):
        """
        Args:
            impath (str): path to image file
            square_size (float): size in meter of a square on the chessboard
        """
        self.im = cv2.imread(impath)
        self.square_size = 0.03
        self.rows = 8  # number of rows in the grid pattern to look for on the chessboard
        self.cols = 6  # number of columns in the grid pattern to look for on the chessboard
        self.im_pts = self.locate_landmark()  # pixel coordinate of chessboard's corners
        self.plane_pts = self.get_landmark_world_coordinate()  # world coordinate of chessboard's corners
        self.H = self.find_homography()
        self.debug = debug
        self.im_name = impath[-5]

    def locate_landmark(self, draw_corners=False):
        """Identify corners on the chessboard such that they form a grid defined by inputted parameters

        Args:
            draw_corners (bool): to draw corners or not
        Return:
            np.ndarray: pixel coordinate of chessboard's corners, shape (self.rows * self.cols, 2)
        """
        # convert color image to gray scale
        gray = cv2.cvtColor(self.im, cv2.COLOR_BGR2GRAY)
        # find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (self.rows, self.cols), None)
        # if found, refine these corners' pixel coordinate & store them
        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), Image.refine_criteria)
            if draw_corners:
                cv2.drawChessboardCorners(self.im, (self.rows, self.cols), corners, ret)
                cv2.imshow('im', self.im)
                cv2.waitKey(0)
                cv2.destroyWindow('im')

        return corners.squeeze()

    def get_landmark_world_coordinate(self):
        """

                * world origin is located at the 1st corner
                * x-axis is from corner 0 to corner 1 till corner 7,
                * y-axis is from corner 0 to corner 8 till corner 40,
                * distance between 2 adjacent corners is self.square_size
        Returns:
            np.ndarray: 3D coordinate of chessboard's corners, shape (self.rows * self.cols, 2)
        """
        pass

    def find_homography(self):
        """

        Return:
            np.ndarray: homography, shape (3, 3)
        """
        # get the normalize transformation
        T_norm_im = normalize_trans(self.im_pts)
        T_norm_plane = normalize_trans(self.plane_pts)

        # normalize image points and plane points
        norm_im_pts = (T_norm_im @ homogenize(self.im_pts).T).T  # shape (n, 3)
        norm_plane_pts = (T_norm_plane @ homogenize(self.plane_pts).T).T  # shape (n, 3)

        # TODO: construct linear equation to find normalized H using norm_im_pts and norm_plane_pts
        Q = None

        # TODO: find normalized H as the singular vector Q associated with the smallest singular value
        H_norm = None

        # TODO: de-normalize H_norm to get H
        H = None
        return H

    def construct_v(self):
        """

        Return:
            np.ndarray: shape (2, 6)
        """
        pass

    def find_extrinsic(self, K):
        """

        Arg:
            K (np.ndarray): camera intrinsic matrix, shape (3, 3)
        Returns:
            tuple[np.ndarray]: Rotation matrix (R) - shape (3, 3), translation vector (t) - shape (3,)
        """
        # NOTE: DO YOUR COMPUTATION OF R & t before the line "if self.debug"
        R = np.eye(3)  # this is just a dummy value, you should overwrite this with your computation
        t = np.zeros(3)  # this is just a dummy value, you should overwrite this with your computation

        if self.debug:
            assert self.plane_pts is not None, "Finish function get_landmark_world_coordinate first"
            # create a matrix of size (npts, 4), each row stores [X, Y, 0, 1] which is the homogeneous 3d coordinate
            # of each corner of the chessboard
            points_3d = np.zeros((self.plane_pts.shape[0], 4))
            points_3d[:, :2] = self.plane_pts
            points_3d[:, -1] = 1.0
            # project those 3d points onto image plane using Equation.(1)
            points_2d = (K @ np.concatenate((R, t.reshape(3, 1)), axis=1) @ points_3d.T).T
            points_2d /= points_2d[:, -1].reshape(-1, 1)
            points_2d = np.floor(points_2d).astype(int)
            # draw a circle around each corner
            im_ = self.im.copy()
            for j in range(points_2d.shape[0]):
                cv2.circle(im_, (points_2d[j, 0], points_2d[j, 1]), 5, (0, 255, 0), -1)
            cv2.imshow(f"img{self.im_name}", im_)

        return R, t


if __name__ == '__main__':
    im_name = "img0.jpg"
    image = Image(f'imgs/{im_name}')
    image.locate_landmark(draw_corners=True)

    _plane_pts = image.get_landmark_world_coordinate()
    print(f"result of get_landmark_world_coordinate for {im_name}:\n{_plane_pts}")
    print("==============================================================\n")

    H = image.find_homography()
    print(f"result of find_homography for {im_name}:\nH = \n{H}")
    print("==============================================================\n")

    # test H
    print("Test the computed H using Equation.(7)")
    h = H.flatten()
    plane_pts = homogenize(image.plane_pts)
    residual_norm = []
    for i in range(image.im_pts.shape[0]):
        lhs = np.zeros((2, 9))  # initialize left-hand side of the Equation (7)
        lhs[0, :3] = plane_pts[i, :]
        lhs[0, -3:] = -image.im_pts[i, 0] * plane_pts[i, :]
        lhs[1, 3:6] = plane_pts[i, :]
        lhs[1, -3:] = -image.im_pts[i, 1] * plane_pts[i, :]

        residual = lhs @ h
        residual_norm.append(np.linalg.norm(residual))
        print('{}: res norm: '.format(i), residual_norm[i])

    print('---')
    print('res_norm min: ', min(residual_norm))
    print('res_norm max should be as close to 0 as possible: ', max(residual_norm))
    print("==============================================================")

    v = image.construct_v()
    print(f"result of construct_v for {im_name}")
    print('v.shape: ', v.shape)
    print("---")
    print(f"v = \n", v)
