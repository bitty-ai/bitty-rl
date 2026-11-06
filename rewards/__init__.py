from rewards.base import Reward
from rewards.binary import BinaryReward
from rewards.partial import PartialReward
from rewards.stepwise import StepwiseReward

REWARDS = {"binary": BinaryReward, "partial": PartialReward, "stepwise": StepwiseReward}

__all__ = ["REWARDS", "BinaryReward", "PartialReward", "Reward", "StepwiseReward"]
