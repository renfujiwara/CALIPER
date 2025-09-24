from frouros.detectors.concept_drift import ADWIN, KSWIN
from typing import Any, Optional, Union
import itertools
from scipy.stats import ks_2samp
import numpy as np

class CustomKSWIN(KSWIN):
    def _update(self, value: Union[int, float], **kwargs: Any) -> None:
        super()._update(value, **kwargs)
        self.update_instances = self.config.num_test_instances


class CustomADWIN(ADWIN):
    def _update(self, value: Union[int, float], **kwargs: Any) -> None:
        # pylint: disable=too-many-locals, too-many-nested-blocks
        # NOTE: Refactor function
        self.num_instances += 1
        self._insert_bucket(value=value)
        self.update_instances = None

        if (
            self.num_instances % self.config.clock == 0  # type: ignore
            and self.width > self.config.min_num_instances
        ):
            flag_reduce_width = True

            while flag_reduce_width:
                flag_reduce_width = False
                flag_exit = False
                w0_instances = 0
                w1_instances = self.width
                w0_total = 0
                w1_total = self.total

                for i in range(len(self.buckets) - 1, -1, -1):
                    if flag_exit:
                        break
                    bucket = self.buckets[i]
                    for j in range(bucket.idx - 1):
                        bucket_size = self._bucket_size(index=i)

                        w0_instances += bucket_size
                        w1_instances -= bucket_size
                        w0_total += bucket.total[j]
                        w1_total -= bucket.total[j]

                        if i == 0 and j == bucket.idx - 1:
                            flag_exit = True
                            break

                        if (
                            w1_instances > self.config.min_window_size  # type: ignore
                            and (
                                w0_instances > self.config.min_window_size  # type: ignore # noqa: E501
                            )
                        ):
                            w0_mean = w0_total / w0_instances
                            w1_mean = w1_total / w1_instances
                            threshold = self._calculate_threshold(
                                w0_instances=w0_instances, w1_instances=w1_instances
                            )
                            if np.abs(w0_mean - w1_mean) > threshold:
                                # Drift detected
                                flag_reduce_width = True
                                self.drift = True
                                self.update_instances = w1_instances

                                if self.width > 0:
                                    w0_instances -= self._delete_bucket()
                                    flag_exit = True
                                    break