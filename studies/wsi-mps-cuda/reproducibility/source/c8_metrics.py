"""Retained C8-CPU-METRIC definitions; independent historical contract, not a new computation."""
import numpy as np, math

def fresh_acc():
    return dict(n=0, absolute_sum=0., difference_squared_sum=0., reference_squared_sum=0.,
                cuda_squared_sum=0., dot=0., max_abs=-1., tile_cos=[], zero_reference=0,
                zero_cuda=0, undefined_cosine=0, worst=[], max_abs_location=None)

def accumulate(s, reference, cuda, coords, offset):
    a = reference.astype(np.float64)
    b = cuda.astype(np.float64)
    d = b-a
    ad = np.abs(d)
    s['n'] += d.size
    s['absolute_sum'] += float(ad.sum())
    s['difference_squared_sum'] += float(np.sum(d*d))
    s['reference_squared_sum'] += float(np.sum(a*a))
    s['cuda_squared_sum'] += float(np.sum(b*b))
    s['dot'] += float(np.sum(a*b))
    loc = np.unravel_index(np.argmax(ad), ad.shape)
    value = float(ad[loc])
    if value > s['max_abs']:
        s['max_abs'] = value
        s['max_abs_location'] = dict(row=offset+int(loc[0]), coordinate=coords[offset+loc[0]].tolist(),
                                     feature=int(loc[1]), reference=float(a[loc]), cuda=float(b[loc]))
    an = np.sqrt(np.sum(a*a, axis=1))
    bn = np.sqrt(np.sum(b*b, axis=1))
    valid = (an > 0) & (bn > 0)
    s['zero_reference'] += int(np.sum(an == 0))
    s['zero_cuda'] += int(np.sum(bn == 0))
    s['undefined_cosine'] += int(np.sum(~valid))
    indices = np.flatnonzero(valid)
    cos = np.sum(a[valid]*b[valid], axis=1)/(an[valid]*bn[valid])
    s['tile_cos'].extend(cos.tolist())
    for j in np.argsort(cos)[:5]:
        i = int(indices[j])
        s['worst'].append(dict(row=offset+i, coordinate=coords[offset+i].tolist(),
                               cosine_raw=float(cos[j]), reference_norm=float(an[i]),
                               cuda_norm=float(bn[i]), tile_mae=float(ad[i].mean()),
                               tile_max_abs=float(ad[i].max())))
    s['worst'] = sorted(s['worst'], key=lambda x: x['cosine_raw'])[:5]

def finish(s):
    raw = np.asarray(s.pop('tile_cos'), dtype=np.float64)
    c = np.clip(raw, -1., 1.)
    denominator = math.sqrt(s['reference_squared_sum']*s['cuda_squared_sum'])
    distribution = None
    if len(c):
        distribution = dict(defined_tiles=len(c), min=float(c.min()), mean=float(c.mean()),
                            std_population=float(c.std()), max=float(c.max()),
                            p01=float(np.percentile(c, 1)), p05=float(np.percentile(c, 5)),
                            median=float(np.median(c)), p95=float(np.percentile(c, 95)),
                            p99=float(np.percentile(c, 99)), raw_min=float(raw.min()),
                            raw_max=float(raw.max()), roundoff_clipped_count=int(np.sum(raw != c)))
    return dict(mae=s['absolute_sum']/s['n'] if s['n'] else None, max_abs=s['max_abs'],
                relative_l2=math.sqrt(s['difference_squared_sum']/s['reference_squared_sum'])
                    if s['reference_squared_sum'] else None,
                global_cosine=float(np.clip(s['dot']/denominator, -1., 1.)) if denominator else None,
                tile_cosine=distribution, zero_norm_reference_tiles=s['zero_reference'],
                zero_norm_cuda_tiles=s['zero_cuda'], undefined_tile_cosine=s['undefined_cosine'],
                worst_five_tile_cosines=s['worst'], max_abs_location=s['max_abs_location'],
                reference_frobenius_norm=math.sqrt(s['reference_squared_sum']),
                cuda_frobenius_norm=math.sqrt(s['cuda_squared_sum']))
