import sys
sys.path.append("/Users/simon/AI-Presentation-Feedback/MultiBench")

# Example: import a dataloader for the MOSI dataset
from datasets.affect.get_data import get_dataloader

# Example: import a fusion model
from fusions.common_fusions import Concat  # if you want to use the Concat fusion

print("✅ Imports successful!")
