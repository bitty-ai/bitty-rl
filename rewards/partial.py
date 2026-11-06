from rewards.base import VALUES_OK, Reward


class PartialReward(Reward):
    format_bonus = 0.1
    call_weight = 0.5

    def score_stages(self, stages):
        correct = sum(s == VALUES_OK for s in stages)
        if correct == len(stages):
            return 1.0
        return self.format_bonus + self.call_weight * correct / len(stages)
