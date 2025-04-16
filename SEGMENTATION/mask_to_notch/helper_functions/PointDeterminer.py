import numpy as np
from helper_functions.mask_utils import get_contpt, get_high_pt

class PointDeterminer:
    def __init__(self, mask, percentage=0.05, degree=25, **kwargs):
        self.mask = mask
        self.percentage = percentage
        self.degree = degree

    @staticmethod
    def fit_circle(y, x):
        x_m = np.mean(x)
        y_m = np.mean(y)

        u = x - x_m
        v = y - y_m

        s_uv = sum(u * v)
        s_uu = sum(u ** 2)
        s_vv = sum(v ** 2)
        s_uuv = sum(u ** 2 * v)
        s_uvv = sum(u * v ** 2)
        s_uuu = sum(u ** 3)
        s_vvv = sum(v ** 3)

        a = np.array([[s_uu, s_uv], [s_uv, s_vv]])
        b = np.array([s_uuu + s_uvv, s_vvv + s_uuv]) / 2.0
        uc, vc = np.linalg.solve(a, b)

        xc_1 = x_m + uc
        yc_1 = y_m + vc

        radius = np.mean(np.sqrt((x - xc_1) ** 2 + (y - yc_1) ** 2))

        return yc_1, xc_1, radius

    def det_side(self):
        lsf = lsf(self.mask)

        if lsf[1][0] > get_high_pt(self.mask)[1]:
            return "left"
        else:
            return "right"

    def correct_angle(self):
        side = self.det_side()

        if side == "left":
            return abs(self.degree)
        else:
            return -abs(self.degree)

    def __call__(self, return_other_pts=False):

        contour = get_contpt(self.mask)
        rotated_y, rotated_x = contour

        indices = np.argsort(rotated_y)[: int(self.percentage * len(rotated_y))]

        yc, xc, radius = self.fit_circle(contour[0][indices], contour[1][indices])

        if return_other_pts:
            return (
                (yc, xc, radius),
                tuple(zip(contour[0][indices], contour[1][indices])),
            )
        return yc, xc, radius