from trainer import trainer
from config import configs
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

config = configs()
runner = trainer(config)
runner.run()