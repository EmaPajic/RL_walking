"""
@author: user
"""

import numpy as np
from rl.processors import WhiteningNormalizerProcessor

class BipedalProcessor(WhiteningNormalizerProcessor):
    def process_action(self, action):
        return np.clip(action, -1., 1.)
