from __future__ import division, absolute_import, print_function

from conf.config import ReinforceConfig
from src.pipeline import run_agent_pipeline

# Set up a virtual display for rendering OpenAI gym environments.
# display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
config = ReinforceConfig()

try:
    run_agent_pipeline(config)
except Exception as e:
    print(e)
