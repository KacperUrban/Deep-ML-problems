def compute_efficiency(n_experts, k_active, d_in, d_out):
    """
    Calculate computational savings of MoE vs. dense layer.

    Args:
        n_experts: Total number of experts
        k_active: Number of active experts (sparsity)
        d_in: Input dimension
        d_out: Output dimension

    Returns:
        Percentage savings in FLOPs
    """
    total_flops = n_experts * d_in * d_out
    experts_flops = k_active * d_in * d_out
    percentage_savings = (total_flops - experts_flops) / total_flops * 100
    return percentage_savings