#attacks/model_poisoning.py
# def sign_flipping(weights):
#     """
#     Reverse gradients → breaks FedAvg
#     """
#     return {k: -1 * v for k, v in weights.items()}

def sign_flipping(weights, scale=5):
    return {k: -scale * v for k, v in weights.items()}


def scaling_attack(weights, scale=5):
    """
    Amplify update → dominates aggregation
    """
    return {k: v * scale for k, v in weights.items()}