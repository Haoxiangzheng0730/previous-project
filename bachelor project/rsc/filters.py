# encoding:utf-8
# kalman filter


class Filter:
    def __init__(self):
        pass

    def put(self, value):
        raise NotImplemented("Not implemented")

    def get(self):
        raise NotImplemented("Not implemented")


class KalmanFilter(Filter):
    def __init__(self, Q, R, one):
        self.estimation = None  # 后验估计
        self.P = one  # 后验估计误差协方差
        self.R = R  # 噪声协方差测量
        self.Q = Q  # 噪声协方差处理
        self.one = one

    def put(self, measured_value):
        if self.estimation is None:
            self.estimation = measured_value
            return self.estimation

        current_estimation = self.estimation
        _P = self.P + self.Q

        kalman_gain = _P / (_P + self.R)
        new_estimation = current_estimation + kalman_gain * (measured_value - current_estimation)
        self.P = (self.one - kalman_gain) * _P

        self.estimation = new_estimation
        return self.estimation

    def get(self):
        return self.estimation


