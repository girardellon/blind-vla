ALPHA = {
    "type0": 0.00,
    "type1": 0.10,
    "type2": 0.25,
    "type3": 0.45,
    "type4": 0.65,
    "type5": 0.85,
    "type6": 1.00,
}

def fuse_embeddings(
    z_tactile_visual,
    z_visual,
    degradation_type,
):

    alpha = ALPHA[degradation_type]

    z = (
        alpha * z_tactile_visual
        + (1.0 - alpha) * z_visual
    )

    return z