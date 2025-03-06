import numpy as np

def kl_divergence(new_policy, old_policy):
    epsilon = 1e-8
    new_policy = np.clip(new_policy, epsilon, 1 - epsilon)
    old_policy = np.clip(old_policy, epsilon, 1 - epsilon)
    return np.sum(old_policy * np.log(old_policy / new_policy))


def clipping(value, epsilon):
    return np.clip(value, 1 - epsilon, 1 + epsilon)


def grpo_objective(rhos, A, pi_theta_old, pi_theta_ref, epsilon=0.2, beta=0.01) -> float:
    """
    Compute the GRPO objective function.

    Args:
        rhos: List of likelihood ratios (p_i) = pi_theta(o_i | q) / pi_theta_old(o_i | q).
        A: List of advantage estimates (A_i).
        pi_theta_old: List representing the old policy probabilities pi_theta_old(o_i | q).
        pi_theta_ref: List representing the reference policy probabilities pi_ref(o_i | q).
        epsilon: Clipping parameter (eps).
        beta: KL divergence penalty coefficient (beta).

    Returns:
        The computed GRPO objective value.
    """
    rhos = np.array(rhos)
    A = np.array(A)
    pi_theta_old = np.array(pi_theta_old)
    pi_theta_ref = np.array(pi_theta_ref)

    term1 = rhos * A
    term2 = clipping(rhos, epsilon) * A

    pi_theta = rhos * pi_theta_old

    pi_theta = pi_theta / np.sum(pi_theta)
    pi_theta_ref = pi_theta_ref / np.sum(pi_theta_ref)
    
    term3 = beta * kl_divergence(pi_theta_ref, pi_theta)

    return np.mean(np.minimum(term1, term2) - term3)
