from rewards.base import VALUES_OK, Reward


class StepwiseReward(Reward):
    format_bonus = 0.1

    def score_stages(self, stages):
        progress = sum(s / VALUES_OK for s in stages) / len(stages)
        return self.format_bonus + (1 - self.format_bonus) * progress
