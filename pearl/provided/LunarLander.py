import numpy as np
from pearl.mask import Mask

class LunarLanderTabularMask(Mask):
    """
    Features: [x_pos, y_pos, x_vel, y_vel, angle, angular_vel, leg1_contact, leg2_contact]
    Actions: [Do nothing, Fire left, Fire main, Fire right]
    """
    def __init__(self):
        super().__init__(4)
        self.action_space = 4
        self.weights = self._define_weights()
        self.last_obs = None

    def _define_weights(self) -> np.ndarray:
        # Base static weights shape: (features, actions)
        w = np.zeros((8, self.action_space), dtype=np.float32)
        w[:, 0] = np.array([0.00495172,  0.01400336,  0.00936291,  0.02579853,  0.01242275, 0.00996473, 0.00653894,  0.01121106])
        w[:, 1] = np.array([ 0.0099617,   0.04634631,  0.15779847,  0.10846883, 0.06118572, 0.05044227, 0.04705729,  0.06875943])
        w[:, 2] = np.array([ 0.04257508, 0.11026585,  0.03461922, 0.23343308, 0.07511852, 0.14107469, 0.00164153,  0.07974701])
        w[:, 3] = np.array([0.04758506,  0.04991618, 0.20178059,  0.09916573,  0.12388149,  0.20148169, 0.04215989, 0.1597175 ])
        # Normalize weights to sum to 1 for each action
        for a in range(self.action_space):
            total = np.sum(w[:, a])
            if total > 0:
                w[:, a] /= total
        return w * 10

    def update(self, obs: np.ndarray):
        # obs shape: (1,8)
        self.last_obs = obs.reshape(-1)

    def compute(self, attr: np.ndarray) -> np.ndarray:
        # attr: (features, actions)
        C, A = attr.shape
        if self.last_obs is None:
            # fallback to static weighting if no state
            return np.sum(self.weights * np.sum(attr, axis=(0,2,3)), axis=0)
        
        scores = np.zeros(A, dtype=np.float32)
        for a in range(A):
            for c in range(C):
                feature_attr = attr[c, a]
                feature_attr = abs(feature_attr)
                feature_attr /= np.linalg.norm(attr[:, a]) if np.linalg.norm(attr[:, a]) > 0 else 1.0
                scores[a] += feature_attr * self.weights[c, a] 
                
        return scores
