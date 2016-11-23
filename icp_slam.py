from __future__ import print_function
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import math
import sys
import datetime
from sklearn.neighbors import KDTree


def odds(p):
    if p == 1:
        return float('inf')
    return p / (1 - p)


def odds_inv(p):
    if p == float('inf'):
        return 1.0
    return p / (1 + p)


def output_pose(pose):
    print('at: %s, rotate: %f deg' %
        (str(pose[:2]), pose[2] / math.pi * 180))


class Map:
    def __init__(self, file):
        self.real_map = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        rows = self.real_map.shape[0]
        cols = self.real_map.shape[1]
        self.x = np.arange(0, cols)
        self.y = np.arange(0, rows)
        self.z = np.ndarray(shape=[rows, cols], dtype=np.float32)
        for i in range(rows):
            for j in range(cols):
                self.z[i][j] = self.real_map[i][j] / 255.0


    def scan_hits(self, pose, max_range, max_radius, sample_count):
        x = pose[0]
        y = pose[1]
        a = pose[2]
        rows = self.y.shape[0]
        cols = self.x.shape[0]
        if x < 0 or x >= cols or y < 0 or y > rows:
            return

        points = list()
        delta = max_radius / sample_count
        start_radius = (2 * math.pi - max_radius) / 2 + math.pi / 2
        for theta in [start_radius + delta * i for i in range(sample_count)]:
            for l in range(max_range):
                dy = l * math.sin(theta)
                dx = l * math.cos(theta)
                newy = int(y + dy)
                newx = int(x + dx)
                if (newx < 0 or newx >= cols or newy < 0 or newy >= rows) or \
                    self.real_map[newy][newx] > 0.5:
                    p = [dx, dy]
                    if p not in points:
                        points.append(p)
                    break
        R = np.array([[math.cos(a), -math.sin(a)],
            [math.sin(a), math.cos(a)]])
        return np.matmul(np.array(points, dtype=np.float32), R)


class Submap():
    def __init__(self, max_scan_range, max_radius):
        self.poses = list()
        self.max_scan_range = max_scan_range
        self.max_radius = max_radius
        self.p_hit = 0.86
        self.p_miss = 1 - self.p_hit


    def get_hits(self):
        hits = list()
        for y in range(self.rows):
            for x in range(self.cols):
                if self.grid[y][x] > self.p_miss + 0.1:
                    hits.append([x, y])
        return np.array(hits)


    def estimate(self, points, pose):
        T = pose[:2]
        a = -pose[2]
        R = np.array([[math.cos(a), -math.sin(a)],
            [math.sin(a), math.cos(a)]])
        return np.matmul(points, R) + T


    def best_fit_transform(self, A, B):
        assert len(A) == len(B)

        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        H = np.matmul(BB.T, AA)
        U, S, V = np.linalg.svd(H)
        R = np.dot(U, V)

        T = centroid_A - np.matmul(R, centroid_B)
        return R, T


    def icp(self, estimate_points, estimate_pose, max_iteration=30):
        correct_hits = self.get_hits()
        kdtree = KDTree(correct_hits, leaf_size=len(correct_hits),
            metric='euclidean')

        r = np.eye(2)
        t = np.zeros(shape=[1, 2])
        B = np.copy(estimate_points)

        k = 3
        best_error = self.max_scan_range
        for it in range(max_iteration):
            # data association
            dist, indices = kdtree.query(B, k=k)
            valid = np.sum(dist / k, axis=1) < best_error * 1.1
            A = correct_hits[indices[valid, 0], :]
            for i in range(1, k):
                A += correct_hits[indices[valid, i], :]
            A /= k
            A = np.reshape(A, [A.shape[0], 2])
            B = B[valid, :]

            if len(B) == 0 or len(A) == 0:
                break

            R, T = self.best_fit_transform(A, B)
            B = np.matmul(B, R) + T
            diff = B - A
            error = np.sum(diff * diff) / diff.shape[0]
            if error < best_error:
                best_error = error
            print('iteration: %d error: %f' % (it, error))
            r = np.matmul(R, r)
            t += T
        pose = np.copy(estimate_pose)
        pose[:2] = np.matmul(pose[:2], r) + t
        pose[2] -= math.asin(r[1][0])
        return r, t, pose


    def setup(self):
        dim = int(self.max_scan_range * 2)
        self.grid = np.zeros(shape=[dim, dim], dtype=np.float32)
        #  self.grid = np.zeros(shape=[300, 300], dtype=np.float32)
        # initial probability of 0.5
        self.grid += self.p_miss
        self.rows = self.grid.shape[0]
        self.cols = self.grid.shape[1]
        self.min_point = np.array([0, 0])
        self.max_point = np.array([dim, dim])


    def update_grid(self, points):
        min_point = np.array([np.min(points[0, :]), np.min(points[1, :])])
        max_point = np.array([np.max(points[0, :]), np.max(points[1, :])])
        ref = np.copy(self.min_point)
        if min_point[0] < self.min_point[0] or min_point[1] < self.min_point[1]:
            ref = (ref - min_point).astype(np.int32)
            self.min_point = np.copy(min_point.astype(np.int32))
        if np.linalg.norm(max_point) > np.linalg.norm(self.max_point):
            self.max_point = np.copy(max_point.astype(np.int32))

        w = self.max_point[0] - self.min_point[0]
        h = self.max_point[1] - self.min_point[1]
        new_grid = np.zeros(shape=[h, w], dtype=np.float32)

        w = self.grid.shape[1]
        h = self.grid.shape[0]
        x = ref[0]
        y = ref[1]

        new_grid[x:x+w, y:y+h] = self.grid
        self.grid = new_grid
        self.rows = self.grid.shape[0]
        self.cols = self.grid.shape[1]

        for point in points:
            x = int(point[0])
            y = int(point[1])
            if x >= 0 and x < self.cols and y >= 0 and y < self.rows:
                self.grid[y][x] = odds_inv(odds(self.grid[y][x]) *
                    odds(self.p_hit))


    def prepare_scan_points(self, raw_points, estimate_pose):
        points = self.estimate(raw_points, estimate_pose)
        R, T, pose = self.icp(points, estimate_pose)
        return R, T, pose


    def update_scan(self, raw_points, pose):
        new_pose = np.copy(pose)
        r = np.eye(2)
        t = np.zeros(shape=[1, 2])
        if len(self.poses) == 0:
            self.setup()
            points = self.estimate(raw_points, pose)
            self.update_grid(points)
        else:
            for i in range(6):
                R, T, pose = self.prepare_scan_points(raw_points, new_pose)
                new_pose = pose
                print('new pose ', end='')
                output_pose(new_pose)
                r = np.matmul(R, r)
                t += T
            points = self.estimate(raw_points, pose)
            self.update_grid(points)
        self.poses.append(pose)
        return r, t, pose


class Robot:
    def __init__(self, p, xi, max_range, max_radius,
            measurement_noise, motion_noise, steering_noise):
        self.p = np.copy(p)
        self.xi = np.copy(xi)
        self.poses = list()
        self.max_range = max_range
        self.max_radius = max_radius
        self.sample_count = int(max_radius / math.pi * 180)
        self.measurement_noise = measurement_noise
        self.motion_noise = motion_noise
        self.steering_noise = steering_noise
        self.submap = Submap(max_range, max_radius)


    def motion_update(self, u, noise=True):
        T = np.array([u[0] * math.cos(u[1]), u[0] * math.sin(u[1])])
        self.xi[2] += u[1]
        if self.xi[2] > 2 * math.pi:
            self.xi[2] -= 2 * math.pi
        self.xi[:2] += T

        if noise:
            translate_noise = np.random.normal(0, self.motion_noise, 2) * T
            rotation_noise = np.random.normal(0, self.steering_noise)
            self.p[:2] = self.xi[:2] + translate_noise
            self.p[2] = self.xi[2] + rotation_noise


    def measurement_update(self, m, output=True):
        raw_points = m.scan_hits(self.p, self.max_range, self.max_radius,
            self.sample_count)
        R, T, new_pose = self.submap.update_scan(raw_points, self.xi)
        print('old pose', end='')
        output_pose(self.xi)
        self.xi = new_pose
        if output:
            print('rotation: %f deg, translation: %s' %
                (math.asin(R[1][0]) / math.pi * 180, str(T)))
            print('self pose: ', end='')
            output_pose(self.xi)
            print('real pose: ', end='')
            output_pose(self.p)


def icp_slam(count, robot, m, motions):
    robot.motion_update(motions[count])
    robot.measurement_update(m)
    display = np.copy(robot.submap.grid)
    image = plt.imshow(display)
    return image,


def main():
    m = Map('./map.png')

    p = np.array([56, 66, 0], dtype=np.float32)
    xi = np.copy(p)
    max_range = 120
    max_radius = (360.0 / 180.0) * math.pi
    measurement_noise = 3
    motion_noise = 2.0
    steering_noise = math.pi / 24

    robot = Robot(p, xi, max_range, max_radius,
        measurement_noise, motion_noise, steering_noise)
    robot.measurement_update(m)

    robot.motion_update(np.array([5, 0], dtype=np.float32))
    robot.measurement_update(m)

    plt.figure()
    plt.imshow(robot.submap.grid)
    plt.colorbar()

    #  motions = []
    #  frames = 12
    #  for i in range(frames):
    #      motions.append(np.array([5, 0], dtype=np.float32))
    #  figure = plt.figure()
    #  anim = animation.FuncAnimation(figure, icp_slam, frames*2,
    #      fargs=(robot, m, motions), interval=500, blit=True)
    #  anim.repeat = False
    plt.show()


if __name__ == '__main__':
    main()
