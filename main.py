from trainer import trainer
from config import configs

config = configs()
runner = trainer(config)
runner.run()