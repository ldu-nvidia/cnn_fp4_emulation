from trainer import reduced_precision_trainer
from config import configs

config = configs()
runner = reduced_precision_trainer(config)
runner.run()