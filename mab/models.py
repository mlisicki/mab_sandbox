### MAB models
import numpy as np
from utils import *

def normalize_reward(reward, reward_history, method="quantile"):
    # 'method' field is a placeholder for future normalization implementations
    if method == "quantile":
        def apply_quantile(reward, q_lo, q_hi):
            reward = (reward - q_lo) / (q_hi - q_lo)
            reward[reward < q_lo] = 0
            reward[reward > q_hi] = 1
            reward[(q_hi - q_lo) == 0] = 0.5
            return reward
        q_lo, q_hi = np.quantile(np.vstack(reward_history), [0.2, 0.8], axis=0)
        #print(reward_history)
        #print("q_lo = {}".format(q_lo))
        #print("q_hi = {}".format(q_hi))
        reward = apply_quantile(reward, q_lo, q_hi)
        last_reward = apply_quantile(reward_history[-1], q_lo, q_hi)
    return reward, last_reward

class GaussianMAB(object):
    def __init__(self, params):
        if not isinstance(params[0], tuple):
            params = [(p,) for p in params]
        # Some of the derived classes need only means
        self.means = np.array([float(p[0]) for p in params])
        if len(params[0]) > 1:
            self.var = np.array([float(p[1]) for p in params])
        self.size = len(params)
        self.active_arms = np.ones(self.size, np.bool) # Used e.g. in zooming algorithm:

    def draw(self, a):
        return np.random.normal(loc=self.means[a], scale=self.var[a])


class NonStationaryGaussianMAB(GaussianMAB):
    def update(self):
        self.means += np.random.normal(loc=0.0, scale=0.01, size=len(self.means))

    def reset(self):
        self.means *= 0.0
        self.active_arms |= True


class NonStationaryTruncatedGaussianMAB(GaussianMAB):
    def __init__(self, params):
        if not isinstance(params[0], tuple):
            params = [(p,) for p in params]
        assert all([p[0] >= 0.1 and p[0] <= 0.9 for p in params])
        if len(params[0]) > 1:
            assert all([p[1] <= 0.5 for p in params])
        super().__init__(params)
        self.orig_means = self.means.copy()

    def draw(self, a):
        r = better_truncnorm.rvs(start=0.0, end=1.0, mean=self.means[a], stdev=self.var[a])
        if (r < 0 or r > 1):
            print(mab.means[a], mab.var[a], r)
        assert r >= 0.0 and r <= 1.0
        return r

    def update(self):
        self.means += np.random.normal(loc=0.0, scale=0.01, size=len(self.means))
        self.means[self.means < 0.1] += 0.1 - self.means[self.means < 0.1]
        self.means[self.means > 0.9] -= self.means[self.means > 0.9] - 0.9
        assert all([m >= 0.1 and m <= 0.9 for m in self.means])

    def reset(self):
        self.means = self.orig_means.copy()


class CorrelatedNonstationaryGaussianMAB(NonStationaryTruncatedGaussianMAB):
    def __init__(self, params, alpha=1, forgetting_rate=0, reward_type="single"):
        super().__init__(params)
        self.startIndex = 1  # position of the first arm on the x axis, when computing correlations
        self.last_action = 0
        self.alpha = alpha  # improvement rate
        # kernel of the improvement function
        self.sigma = lambda x: np.abs(x) * 0.2
        self.kern = lambda x, mu: np.exp(-0.5 * ((x - mu) / (self.sigma(mu))) ** 2) / (
                    np.sqrt(2 * np.pi) * self.sigma(mu))
        self.forgetting_rate = forgetting_rate
        self.active_arms &= False
        self.reward_type = reward_type
        self.last_means = self.means.copy()
        self.best_means = self.means.copy()
        self.reward_history = []

    def get_context(self):
        # Set the same context for each arm and let it learn the appropriate reward function for each action
        return [self.means]*len(self.means)

    @property
    def var(self):
        '''
        This represents being more confident about the score of the model as it improves
        '''
        return (0.9 * self.means + 0.1) * 0.05

    def draw(self, a):
        self.last_action = a
        if self.reward_type == "single":
            return super().draw(a)
        elif self.reward_type == "mean":
            return 1-self.means.mean()
        elif self.reward_type == "mean2":
            output = (self.last_means-self.means).mean()
            self.last_means = self.means.copy()
            return output
        elif self.reward_type == "best":
            output = ((self.best_means-self.means).mean()+1)/2
            self.best_means = np.min([self.best_means, self.means], axis=0)
            return output
        elif self.reward_type == "best2":
            diff = self.best_means - self.means
            output = diff[diff>=0].mean() if len(diff[diff>=0])>0 else 0
            self.best_means = np.min([self.best_means, self.means], axis=0)
            return output
        elif self.reward_type == "best3":
            diff = (self.best_means - self.means)

            diff[diff>=0]*=10
            diff[diff>1]=1
            diff[diff<-1]=-1
            output = (diff.mean()+1)/2
            self.best_means = np.min([self.best_means, self.means], axis=0)
            return output
        elif self.reward_type == "best_quantile":
            raw_reward = (self.best_means - self.means).mean()
            if self.reward_history:
                reward, _ = normalize_reward(raw_reward, self.reward_history)
            else:
                reward = 0.5
            self.reward_history += [[raw_reward]]
            self.best_means = np.min([self.best_means, self.means], axis=0)
            return reward
        elif self.reward_type == "best_quantile2":
            diff = (self.best_means - self.means)
            raw_reward = diff[diff>=0].sum()
            if self.reward_history:
                reward, _ = normalize_reward(raw_reward, self.reward_history)
            else:
                reward = 0.5
            self.reward_history += [[raw_reward]]
            self.best_means = np.min([self.best_means, self.means], axis=0)
            return reward
        else:
            raise Exception("Wrong reward type")

    def update(self):
        '''
        Update the distribution of arms based on the most recently pulled arms
        '''
        self.means -= self.alpha * self.kern(np.arange(self.startIndex, len(self.means) + self.startIndex, 1),
                                             self.last_action + self.startIndex)
        self.means += self.forgetting_rate
        self.means[self.means < 0.1] = 0.1
        self.means[self.means > 0.9] = 0.9
        assert all([m >= 0.1 and m <= 0.9 for m in self.means])

    def reset(self):
        self.means = self.orig_means.copy()
        self.last_means = self.means.copy()
        self.best_means = self.means.copy()
        self.reward_history = []
