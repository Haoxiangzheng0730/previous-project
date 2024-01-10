import cv2
import math
import random
import numpy as np


def transform(mat, point):
    return (
        mat[0, 0] * point[0] + mat[0, 1] * point[1] + mat[0, 2],
        mat[1, 0] * point[0] + mat[1, 1] * point[1] + mat[1, 2]
    )


class RansacMotionEstimator:
    def __init__(self, max_iterations, max_distance, hypothesis_set_length=10, min_inliers=40,
                 remember_inlier_indices=True):
        self.hypothesis_set_length = hypothesis_set_length
        self.max_iterations = max_iterations
        self.max_distance = max_distance
        self.min_inliers = min_inliers
        self.remember_inlier_indices = remember_inlier_indices

    def estimate(self, first_set, second_set):
        set_length = len(first_set)
        best_model = None
        best_model_inliers = 0
        inliers_indices = []
        outliers_indices = []
        range_set_length = range(set_length)

        for _ in range(self.max_iterations):
            if set_length < 3:
                continue
            random_indices = None
            if set_length > self.hypothesis_set_length:
                # 返回从0-set_length内选择的hypothesis_set_length列表
                random_indices = random.sample(range_set_length, self.hypothesis_set_length)
            else:
                random_indices = range_set_length

            # 创建随机子集
            first_subset = np.array([first_set[i] for i in random_indices], dtype=np.float32)
            second_subset = np.array([second_set[i] for i in random_indices], dtype=np.float32)

            # 随机子集的estimate model
            current_model = cv2.estimateRigidTransform(first_subset, second_subset, fullAffine=True)

            if current_model is None:
                continue

            current_model_inliers = 0
            current_inliers_indices = []
            current_outliers_indices = []

            for index in range(set_length):
                transformed_point = transform(current_model, first_set[index][0])
                error = math.sqrt(
                    math.pow(transformed_point[0] - second_set[index][0][0], 2)
                    + math.pow(transformed_point[1] - second_set[index][0][1], 2)
                )

                if error < self.max_distance:
                    current_model_inliers += 1
                    if self.remember_inlier_indices:
                        current_inliers_indices.append(index)
                else:
                    current_outliers_indices.append(index)

            if current_model_inliers > best_model_inliers:
                best_model = current_model
                best_model_inliers = current_model_inliers
                inliers_indices = current_inliers_indices
                outliers_indices = current_outliers_indices

        if best_model is None or (self.min_inliers is not None and best_model_inliers < self.min_inliers):
            best_model = cv2.estimateRigidTransform(first_set, second_set, fullAffine=True)
            if self.remember_inlier_indices:
                inliers_indices = [i for i in range(set_length)]

        return best_model, inliers_indices, outliers_indices
