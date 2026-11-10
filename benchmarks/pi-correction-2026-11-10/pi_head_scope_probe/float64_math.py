"""NumPy float64 reconstruction from saved FP32 plan arrays."""
import numpy as np


def reconstruct(arrays, prefix):
    def get(k): return arrays[prefix + '_' + k].astype(np.float64)
    mu, logstd, target, observed, wheelbase = [get(k) for k in ('mu', 'logstd', 'native_target', 'observed', 'wheelbase')]
    times = np.array([(i / 32.)**2 * 10 for i in range(33)], dtype=np.float64)
    tw = np.array([1. if t <= 4 else max(.2, 1. - .8 * (t - 4.) / 6.) for t in times], dtype=np.float32).astype(np.float64)
    cw = np.array([1., 1., .1, .5, .5, .1, .1, .1, 0., 0., 0., .5, 0., 0., .3], dtype=np.float32).astype(np.float64)
    w = tw[None, :, None] * cw[None, None, :]
    log_b = np.clip(logstd, -7., 7.)
    nll = np.mean(np.sum((np.abs(target - mu) * np.exp(-log_b) + log_b) * w, axis=(1, 2))) / w.sum()
    n = int(np.searchsorted(times, 1., side='right'))
    dt = np.diff(times[:n])
    def residual(plan):
        secant = (plan[:, 1:n, 3:5] - plan[:, :n-1, 3:5]) / dt[None, :, None]
        return np.stack((plan[:, :n-1, 6:8] - secant, plan[:, 1:n, 6:8] - secant), axis=2)
    err = np.abs(residual(mu) - residual(target))
    consistency = np.mean(np.where(err < 1., .5 * err**2, err - .5))
    def action(plan):
        def interp(x, t):
            hi = int(np.searchsorted(times, t))
            a = (t - times[hi-1]) / (times[hi] - times[hi-1])
            return x[:, hi-1] * (1-a) + x[:, hi] * a
        sp = np.maximum(observed[:, 0], 1.)
        steer = np.arctan(wheelbase * (-2. * interp(plan[:, :, 11], .2) / (sp * .2) + plan[:, 0, 14] / sp))
        speed = np.linalg.norm(plan[:, :, 3:5], axis=-1)
        tangent = (plan[:, :, 3:5] * plan[:, :, 6:8]).sum(-1) / np.maximum(speed, .001)
        target_speed = interp(speed, .3)
        raw = 1.5 * (target_speed - np.linalg.norm(observed[:, :2], axis=-1)) + .5 * interp(tangent, .3)
        return np.stack((steer, raw, target_speed, interp(tangent, .3)), axis=-1)
    predicted, native = action(mu), action(target)
    error = np.abs(predicted - native)
    huber = lambda values: float(np.mean(np.where(values < .05, .5 * values**2 / .05, values - .025)))
    losses = {'plan_nll': float(nll), 'native_relative_acceleration': float(consistency),
              'plan_l1': float(np.mean(np.sum(np.abs(mu-target)*w, axis=(1, 2))) / w.sum()),
              'steer': huber(error[:, 0]), 'longitudinal': huber(error[:, 1])}
    return {'losses': losses, 'objective': float(nll+consistency), 'predicted': predicted,
            'native': native, 'absolute_action_error': error}

