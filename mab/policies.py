import numpy as np
from utils import *
machine_eps = np.finfo(float).eps

# Policies
def adaptive_zooming1(mab, Q, N, t, delta, active_set):
    uncertainty = np.sqrt(2 * np.log(1 / delta(t + 1)) / (N + 1))

    covered = []
    for i in range(mab.size):
        if mab.active_arms[i]:
            covered += list(range(int(i - uncertainty[i] * mab.size), int(i + uncertainty[i] * mab.size) + 1))
    uncovered = [i for i in range(mab.size) if i not in covered]
    if uncovered:
        activated_arm = np.random.choice(uncovered)
        mab.active_arms[activated_arm] = True

    # selection rule
    _ucb = np.full(Q.shape, np.inf)
    _ucb = Q + 2 * uncertainty
    _ucb[~mab.active_arms] = -np.inf
    action = np.argmax(_ucb)
    reward = mab.draw(action)
    N[action] += 1
    Q[action] += 1 / N[action] * (reward - Q[action])
    policy_monitor = uncertainty
    return Q, N, action, reward, policy_monitor


# Adaptive zooming 2 is just a restricted version of the 3rd version of the algorithm

def adaptive_zooming3(mab, Q, N, t, alpha, beta, active_set):
    if t == 0:
        mab.active_arms &= False
    uncertainty = np.sqrt(1 / (alpha * N + 1))

    # activation rule
    covered = []
    for i in range(mab.size):
        if mab.active_arms[i]:
            covered += list(range(int(i - uncertainty[i] * mab.size), int(i + uncertainty[i] * mab.size) + 1))
    uncovered = [i for i in range(mab.size) if i not in covered]
    if uncovered:
        activated_arm = np.random.choice(uncovered)
        mab.active_arms[activated_arm] = True

    # selection rule
    _ucb = Q + beta * uncertainty
    _ucb[~mab.active_arms] = -np.inf
    action = np.argmax(_ucb)
    reward = mab.draw(action)
    N[action] += 1
    Q[action] += 1 / N[action] * (reward - Q[action])
    policy_monitor = uncertainty
    return Q, N, action, reward, policy_monitor


def epsilonGreedy(mab, Q, N, t, epsilon, alpha):
    if np.random.rand() <= epsilon:
        action = np.random.choice(mab.size)
    else:
        action = np.argmax(Q)
    reward = mab.draw(action)
    N[action] += 1
    Q[action] += alpha(N[action]) * (reward - Q[action])
    return Q, N, action, reward, None


def ucb(mab, Q, N, t, delta):
    _ucb = np.full(Q.shape, np.inf)
    _ucb[N > 0] = Q[N > 0] + np.sqrt(2 * np.log(1 / delta(t + 1)) / N[N > 0])
    action = np.random.choice(np.argwhere(_ucb == np.amax(_ucb)).flatten())
    reward = mab.draw(action)
    N[action] += 1
    Q[action] += 1 / N[action] * (reward - Q[action])
    return Q, N, action, reward, None


def lin_ucb(mab, Q, N, t, alpha):
    mab_size = len(Q)
    context = mab.get_context()

    _ucb = np.full(mab_size, np.inf)
    for i in range(len(Q)):
        if Q[i] is None:
            continue
        A, b = Q[i]
        A_inv = np.linalg.inv(A)
        theta = np.dot(A_inv, b)
        _ucb[i] = np.dot(context[i], theta) + alpha * np.sqrt(np.dot(np.dot(context[i], A_inv), context[i].T))

    action = np.random.choice(np.argwhere(_ucb == np.amax(_ucb)).flatten())
    reward = mab.draw(action)
    N[action] += 1
    if Q[action] is None:
        A = np.identity(mab_size)
        b = np.zeros((mab_size, 1))
    else:
        A, b = Q[action]
    A += np.outer(context[action], context[action])
    b += reward * context[action][:,np.newaxis]
    Q[action] = [A, b]
    return Q, N, action, reward, None


def ucb_slivkins_upfal(mab, Q, N, t, sigma=0.01):
    # sigma is defined in the non-stationary bandit class
    _ucb = np.full(Q.shape, np.inf)
    if t>0:
        _ucb[N > 0] = Q[N > 0] + np.sqrt(2 * np.log(t) / N[N > 0]) + sigma * np.sqrt(8 * t * np.log(t))
    action = np.random.choice(np.argwhere(_ucb == np.amax(_ucb)).flatten())
    reward = mab.draw(action)
    N[action] += 1
    Q[action] += 1 / N[action] * (reward - Q[action])
    return Q, N, action, reward, None

def exp3(mab, w, N, t, eta, epsilon=0):
    '''
    Following Graves (Exp3.S below) we can decouple the two parameters: epsilon (aka gamma in Auer)
    and eta (as in Lattimore). In Auer's Exp3 paper there is just one coupled gamma parameter. I couldn't
    get that version to produce good results in my application. The current version is more flexible and allows for
    reverting to the original version by setting eta=epsilon.
    '''
    if all(w == 0):
        w.fill(0)
    P = (1 - epsilon) * np.exp(w) / np.sum(np.exp(w)) + epsilon / mab.size
    P /= P.sum()
    action = np.random.choice(mab.size, p=P)
    reward = mab.draw(action)
    w[action] += eta * reward / (P[action]+machine_eps)
    w -= w.max() + 1
    return w, None, action, reward, None

def exp3_loss_est(mab, w, N, t, eta, epsilon=0):
    if all(w == 0):
        w.fill(0)
    P = (1 - epsilon) * np.exp(w) / np.sum(np.exp(w)) + epsilon / mab.size
    P /= P.sum()
    action = np.random.choice(mab.size, p=P)  # sample an action
    reward = mab.draw(action)  # draw a reward
    w += eta
    w[action] -= eta * (1 - reward) / (P[action]+machine_eps) # update the preference
    w -= w.max() + 1 # Prevent overflow
    return w, None, action, reward, None

def exp3_s(mab, w, N, t, eta, epsilon, beta, estimator="loss"):
    '''
    As described in Automated Curriculum Learning by Graves.
    This is the same as Exp3.S from Auer except from the alpha parameter
    In Exp3.S the weight update takes is based partially on the values of
    all the other weights.
    '''
    # initial weights should be all 0
    alpha = 1 / (t + 1)
    P = (1 - epsilon) * np.exp(w) / np.sum(np.exp(w)) + epsilon / mab.size
    P /= P.sum()
    action = np.random.choice(mab.size, p=P)
    rewards = np.full(mab.size, float(beta))
    reward = mab.draw(action)
    rewards[action] += reward
    if estimator == "reward":
        rw_hat = convert_to_large_num(np.exp(w + eta * rewards / P))
    elif estimator == "loss":
        rw_hat = convert_to_large_num(np.exp(w + eta * (1 + (1 - rewards) / P)))
    else:
        raise Exception("estimator must be either 'reward' or 'loss'")
    # I don't think it's possible to take this term outside of the sum below and get rid of the
    # exponential to improve numerical stability.
    w = np.log((1 - alpha) * rw_hat + \
        (alpha / (mab.size - 1)) * np.array([np.sum(rw_hat[np.arange(len(rw_hat)) != i]) for i in range(len(rw_hat))]))
    w = convert_to_large_num(w)
    w -= w.max() + 1
    return w, None, action, reward, None
