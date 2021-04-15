import numpy as np

def run(mab, policy, steps=1000, **policy_params):
    if hasattr(mab, 'reset'):
        mab.reset()
    if policy.__name__ == "lin_ucb":
        Q = [None]*mab.size
    else:
        Q = np.zeros(mab.size).astype('float')
    N = np.zeros(mab.size)
    monitor = np.zeros((steps, 6 + mab.size*3))
    for t in range(steps):
        Q, N, action, reward, policy_monitor = policy(mab, Q, N, t, **policy_params)
        if policy_monitor is None: policy_monitor = []
        policy_monitor = np.pad(policy_monitor,(0,mab.size-len(policy_monitor)))
        opt_action = np.argmax(mab.means)
        monitor[t, :] = [action, reward, 0, opt_action, 0, 0, *mab.means,
                         *mab.active_arms, *policy_monitor]
        if hasattr(mab, 'update'):
            mab.update()
    monitor[:, 2] = np.cumsum(monitor[:, 1]) / np.arange(1, steps + 1)  # average reward
    # Update this after modifying monitor. Currently mab.means start at index 6 in the monitor
    means_start_idx = 6
    monitor[:, 4] = np.max(monitor[:, means_start_idx:means_start_idx+mab.size], axis=1) - \
                    monitor[np.arange(len(monitor)), (means_start_idx + monitor[:, 0]).astype(
                        'int')]  # regret = max_mu - chosen_mu
    monitor[:, 5] = np.cumsum(monitor[:, 4])  # cumulative pseudo-regret
    return monitor
