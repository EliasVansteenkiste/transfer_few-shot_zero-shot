from unittest import TestCase
from mock import MagicMock

from hibashi.data.samplers import MultiIndexBalancingSampler


class TestMultiIndexBalancingSampler(TestCase):

    def test_3_ds(self):
        """
        Test with a collection of three datasets:
            first dataset has a length of 10
            second dataset has a length of 30
            third dataset is an infinite dataset (e.g. random vector)
        The MultiIndexBalancingSampler is configured to draw:
            1 sample from the first dataset
            2 samples from the second dataset
            1 sample from third dataset
        per iteration step.
        """
        collection = MagicMock()
        collection.__len__.return_value = 3

        ds_0 = MagicMock()
        ds_1 = MagicMock()
        ds_2 = MagicMock()

        ds_0.__len__.return_value = 10
        ds_0.finite = True
        ds_1.__len__.return_value = 30
        ds_1.finite = True
        ds_2.finite = False

        assert ds_0.finite
        assert ds_1.finite
        assert not ds_2.finite

        collection.ds = [ds_0, ds_1, ds_2]

        sampler = MultiIndexBalancingSampler(collection, (1, 2, 1), replacement=False)

        idcs_ds1 = []
        idcs_ds2 = []
        idcs_ds3 = []

        for idcs in sampler:
            idcs_ds1.append(idcs[0][0])
            idcs_ds2.append(idcs[1][0])
            idcs_ds2.append(idcs[1][1])
            idcs_ds3.append(idcs[2][0])

        assert len(set(idcs_ds1)) == 10
        assert len(set(idcs_ds1[:10])) == 10
        assert len(set(idcs_ds2)) == 30
        assert len(set(idcs_ds3)) == len(idcs_ds3)
        assert len(list(sampler)) == 15


