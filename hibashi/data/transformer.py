import numpy as np
import torch


class ToTensor(object):
    def __init__(self, methods, strict=True):
        self.methods = methods
        self.strict = strict

    def __call__(self, sample):
        for m_key, method in self.methods.items():
            if m_key not in sample:
                if self.strict:
                    raise ValueError('Key {} not in the sample'.format(m_key))
                else:
                    continue
            else:
                s_value = sample[m_key]
                if method == 'transpose_from_numpy':
                    if not isinstance(s_value, np.ndarray):
                        raise ValueError(
                            'Value type for key {} should be np.ndarray but is {}'.format(m_key, type(s_value)))
                    s_value = s_value.transpose((2, 0, 1))
                    sample[m_key] = torch.from_numpy(s_value)
                elif method == 'from_numpy':
                    sample[m_key] = torch.from_numpy(s_value)
                elif method == 'from_numpy_per_item':
                    sample[m_key] = [torch.from_numpy(item) for item in s_value]
                elif method == 'tensor':
                    sample[m_key] = torch.tensor([s_value])
                else:
                    raise NotImplementedError

        return sample


class ToLongTensor(object):
    def __init__(self, names, per_item=False):
        self.names = names
        self.per_item = per_item

    def __call__(self, sample):
        for name in self.names:
            if self.per_item:
                sample[name] = [item.long() for item in sample[name]]
            else:
                sample[name] = sample[name].long()
        return sample


class ToFloatTensor(object):
    def __init__(self, names, per_item=False, strict=True):
        self.names = names
        self.per_item = per_item
        self.strict = strict

    def __call__(self, sample):
        for name in self.names:
            if not self.strict and name not in sample:
                continue
            if self.per_item:
                sample[name] = [item.float() for item in sample[name]]
            else:
                sample[name] = sample[name].float()
        return sample


class NormalizeSample(object):
    def __init__(self, bounds):
        self.bounds = bounds

    def __call__(self, samples):
        for s_key, s_value in samples.items():
            bound = self.bounds.get(s_key)
            if bound:
                samples[s_key] = (s_value - bound[0]) / (bound[1] - bound[0])

        return samples
