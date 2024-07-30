import math

import numpy as np


class MlfMotionFilter:
    def __init__(self):
        self.EPSION_TIME = 1e-3
        self.DEFAULT_FPS = 0.1

        # switch for filter strategies
        self.use_adaptive = True
        self.use_breakdown = True
        self.use_convergence_boostup = True
        # default covariance parameters for kalman filter
        self.init_velocity_variance = 5.0
        self.init_acceleration_variance = 10.0
        self.measured_velocity_variance = 0.4
        self.predict_variance_per_sqrsec = 50.0
        # other parameters
        self.boostup_history_size_minimum = 3
        self.boostup_history_size_maximum = 6
        self.converged_confidence_minimum = 0.5
        self.noise_maximum = 0.1
        self.trust_orientation_rage = 40

    def state_gain_adjustment(self, track_data, last_obj, crt_obj, state_gain):
        if self.use_adaptive:
            state_gain = crt_obj.update_quality

        # breakdown the constrained the max allowed change of state
        if self.use_breakdown:
            velocity_breakdown_threshold = 10.0 - (track_data.age - 1) * 2
            velocity_breakdown_threshold = max(velocity_breakdown_threshold, 0.3)
            velocity_gain = np.linalg.norm(state_gain[:2])
            if velocity_gain > velocity_breakdown_threshold:
                state_gain[:2] *= velocity_breakdown_threshold / velocity_gain

            #  acceleration breakdown threshold
            acceleration_breakdown_threshold = 2.0
            acceleration_gain = np.linalg.norm(state_gain[-2:])
            if acceleration_gain > acceleration_breakdown_threshold:
                state_gain[-2:] *= acceleration_breakdown_threshold / acceleration_gain
        return state_gain

    def update_with_partial_observation(self, track_data, last_obj, crt_obj):
        dist = np.linalg.norm(crt_obj.get_center())
        last_state = last_obj.get_predict_state()
        last_state_covariance = last_obj.get_state_covariance()

        time_diff = crt_obj.get_ts() - last_obj.get_ts()

        transition = np.eye(4)
        transition[0, 2] = transition[1, 3] = time_diff

        # composition with rotation
        if crt_obj.get_category() not in ["Person", "Pedestrian"] and dist < self.trust_orientation_rage:
            crt_dir = [np.cos(crt_obj.get_utm_yaw()), np.sin(crt_obj.get_utm_yaw())]
            last_dir = [np.cos(last_state.get_utm_yaw()), np.sin(last_obj.get_utm_yaw())]
            crt_dir /= np.linalg.norm(crt_dir)
            last_dir /= np.linalg.norm(last_dir)
            cos_theta = np.dot(crt_dir, last_dir)
            sin_theta = last_dir[0] * crt_dir[1] - last_dir[1] * crt_dir[0]
            rot = np.array(
                [
                    [cos_theta, -sin_theta],
                    [sin_theta, cos_theta]
                ]
            )
            rot_extend = np.zeros((4, 4))
            rot_extend[:2, :2] = rot
            rot_extend[2:, 2:] = rot
            transition = rot_extend * transition

        measurement_covariance = crt_obj.get_measurement_covariance()[:2, :2]

        # 1. prediction stage
        predict_covariance = np.eye(4) * self.predict_variance_per_sqrsec * time_diff * time_diff
        state = transition * last_state
        state_covariance = transition * last_state_covariance * transition.T + predict_covariance

        # 2. measurement update stage
        measurement = crt_obj.selected_measured_velocity[:2]
        direction = crt_obj.get_utm_direction()[:2]
        direction /= np.linalg.norm(direction)
        odirection = np.array([direction[1], -direction[0]])
        if crt_obj.get_category() in ["Person", "Pedestrian"] and dist < self.trust_orientation_rage:
            measurement_covariance = np.eye(2)
            measurement_covariance *= self.measured_velocity_variance
        else:
            kVarianceAmplifier = 9.0
            measurement_covariance = self.measured_velocity_variance * direction * direction.T + \
                                     (self.measured_velocity_variance + abs(measurement.dot(odirection)) *
                                      kVarianceAmplifier) * odirection * odirection.T
        observation_transform = np.zeros((2, 4))
        observation_transform[:2, :2] = np.eye(2)
        kalman_gain_matrix = state_covariance * observation_transform.T * \
                             np.linalg.inv((observation_transform * state_covariance * observation_transform.T + measurement_covariance))
        state_gain = kalman_gain_matrix * (measurement - observation_transform * state)

        # 3. gain adjustment and estimate posterior
        state_gain = self.state_gain_adjustment(track_data, last_obj, crt_obj, state_gain)
        state = state + state_gain
        state_covariance = (np.eye(4) - kalman_gain_matrix * observation_transform) * state_covariance

        # 4. state to belief and output to keep consistency
        crt_obj.belief_velocity_gain = np.array([state_gain[0], state_gain[1], 0])
