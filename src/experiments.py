from train import train

train(name="default_noDrop")
train(name="default_drop_0.2", drop_prob=0.2)
train(name="default_drop_0.5", drop_prob=0.5)
train(name="default_drop_0.8", drop_prob=0.8)
