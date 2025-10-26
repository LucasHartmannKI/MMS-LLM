import numpy as np
from PointLLM.pointllm.data.utils import pc_normalize

def test_pc_normalize_unit_sphere():
    points = np.array([[0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0]], dtype=np.float32)
    normed = pc_normalize(points.copy())
    # centroid should be near zero
    assert np.allclose(normed.mean(axis=0), 0.0, atol=1e-6)
    # all points should lie within unit sphere
    radius = np.sqrt((normed ** 2).sum(axis=1)).max()
    assert np.isclose(radius, 1.0, atol=1e-6)

