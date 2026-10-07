import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC, SKELETON_PEDREC_JOINT_COLORS, SKELETON_PEDREC_LIMB_COLORS
from pedrec.models.human import Human
from pedrec.utils.skeleton_helper import get_joint_score


def draw_skeleton(img: np.ndarray, skeleton: np.ndarray, min_joint_score: float = 0.5):
    """
    Visualizes joints with the given color and radius.
    :param img: The original image
    :param skeletons: The skeleton joints to be visualized, should be in MS COCO format
    """
    for idx, joint in enumerate(skeleton):
        if get_joint_score(joint) > min_joint_score:  # check score
            x_coord, y_coord = int(joint[0]), int(joint[1])
            cv2.circle(img, (x_coord, y_coord), 4, SKELETON_PEDREC_JOINT_COLORS[idx].tuple_bgr, 2)
    for idx, pair in enumerate(SKELETON_PEDREC):
        joint_a = skeleton[pair[0]]
        joint_b = skeleton[pair[1]]
        if (get_joint_score(joint_a) + get_joint_score(joint_b)) / 2 > min_joint_score:
            cv2.line(img, (int(joint_a[0]), int(joint_a[1])), (int(joint_b[0]), int(joint_b[1])),
                     SKELETON_PEDREC_LIMB_COLORS[idx].tuple_bgr,
                     thickness=3)


def get_orientation_plot(theta_rad: float, phi_rad: float):
    phi2d = np.linspace(0, 359, num=360)
    theta2d = np.linspace(0, 179, num=180)
    test = np.meshgrid(phi2d, theta2d)
    phi2d = test[0]
    theta2d = test[1]
    power2d = np.zeros((180, 360), dtype=np.float32)

    phi = int(np.rad2deg(phi_rad))
    theta = int(np.rad2deg(theta_rad))
    power2d[theta - 15:theta + 15, phi - 15:phi + 15] = 1

    THETA = np.deg2rad(theta2d)
    PHI = np.deg2rad(phi2d)
    R = power2d
    Rmax = np.max(R)

    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)

    fig = plt.figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.grid(False)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.plot_surface(

        X, Y, Z, rstride=10, cstride=10, cmap=plt.get_cmap('jet'),

        linewidth=0, antialiased=False, alpha=0.5, zorder=0.5)

    ax.view_init(azim=270, elev=20)

    phi, theta = np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 40)
    PHI, THETA = np.meshgrid(phi, theta)
    R = Rmax
    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)
    ax.plot_wireframe(X, Y, Z, linewidth=0.5, rstride=3, cstride=3)

    canvas.draw()
    img = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


def draw_orientation(img: np.ndarray, human: Human):
    # TODO: get hip middle joint
    # TODO: Draw 3d coordinate system in hip based on phi, theta
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_pose/py_pose.html

    # joint_hip_r = human.skeleton[SKELETON_PEDREC_JOINT.right_hip.value]
    orientation_img = get_orientation_plot(human.orientation[0, 0], human.orientation[0, 1])

    img[0:orientation_img.shape[0], 0:orientation_img.shape[1]] = orientation_img

    # cv2.putText(img, f"{human.orientation[0, 1]*(180/math.pi)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, color=(0, 0, 255))
    # target_y = joint_hip_r[1] + (0 * math.cos(phi) + 50 * math.sin(phi))
    # y_y = y
    # target_x = joint_hip_r[0] + (-0 * math.sin(phi) + 50 * math.cos(phi))
    # cv2.line(img, (int(joint_hip_r[0]), int(joint_hip_r[1])), (int(target_x), int(target_y)), (0, 255, 0))
