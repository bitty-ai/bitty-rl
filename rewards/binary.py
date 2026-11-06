from rewards.base import VALUES_OK, Reward


class BinaryReward(Reward):
    def score_stages(self, stages):
        return float(all(s == VALUES_OK for s in stages))
