from .igt_agent import Agent
from .pvl_delta import PVLDeltaAgent
from .orl_agent import ORLAgent
from .aif_agent import AIFAgent

REGISTRY = {
    "Agent": Agent,
    "PVLDeltaAgent": PVLDeltaAgent,
    "ORLAgent": ORLAgent,
    "AIFAgent": AIFAgent,
}
