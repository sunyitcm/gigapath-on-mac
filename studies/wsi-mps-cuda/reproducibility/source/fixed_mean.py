"""Public extraction of the recorded CPU FP64 mean loop; not a historical whole-file copy."""
def fixed_mean(features, expected_n, expected_d):
    import numpy as np
    if features.shape!=(expected_n,expected_d) or expected_n<=0 or features.dtype!=np.float32:raise ValueError('Complete FP32 matrix required')
    acc=np.zeros(expected_d,np.float64)
    for i in range(0,expected_n,256):
        chunk=features[i:i+256].astype(np.float64)
        if not np.isfinite(chunk).all():raise ValueError('Nonfinite feature')
        acc+=chunk.sum(axis=0)
    return acc/expected_n
