from train import train


train(batch_size=16, name="bonus_noDrop_freeze", drop_prob=0., bonus=True, freeze=True)
train(batch_size=16, name="bonus_drop_0.2_freeze", drop_prob=0.2, bonus=True, freeze=True)
train(batch_size=16, name="bonus_drop_0.5_freeze", drop_prob=0.5, bonus=True, freeze=True)
train(batch_size=16, name="bonus_drop_0.8_freeze", drop_prob=0.8, bonus=True, freeze=True)

train(batch_size=16, name="default_noDrop_freeze", drop_prob=0., freeze=True)
train(batch_size=16, name="default_drop_0.2_freeze", drop_prob=0.2, freeze=True)
train(batch_size=16, name="default_drop_0.5_freeze", drop_prob=0.5, freeze=True)
train(batch_size=16, name="default_drop_0.8"_freeze, drop_prob=0.8, freeze=True)
