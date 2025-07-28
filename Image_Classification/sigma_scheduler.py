from typing import Any, Dict


class SigmaScheduler():
    def __init__(self, sigma_min, sigma_max, n_iteration, gamma=1.0):
        assert sigma_max >= sigma_min, "sigma_max must larger or equal to sigma_min"

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.n_iteration = n_iteration
        self.gamma = gamma
        self.iter = 0
        self.curr_sigma = sigma_max

    def state_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)

    def get_sigma(self):
        return max(self.curr_sigma, self.sigma_min)

    def step(self):
        t = min(self.iter / self.n_iteration, 1.0)

        self.curr_sigma = self.sigma_max - (self.sigma_max - self.sigma_min) * (t ** self.gamma)
        self.iter += 1
