import torch
import os

src = '/home/zhangtao19/trained_weights'
drc = '/home/zhangtao19/trained_weights_new'

names = os.listdir(src)
src_weights = [os.path.join(src, name) for name in names]
drc_weights = [os.path.join(drc, name) for name in names]

for i, weight in enumerate(src_weights):
    weight = torch.load(weight)

    model_ = {}
    for key in weight['model'].keys():
        if 'offline_tracker' in key:
            key_ = key.replace("offline_tracker", "refiner")
        else:
            key_ = key
        print(key)
        print(key_)
        print('-------------------------')
        model_.update({key_: weight['model'][key]})

    weight['model'] = model_
    torch.save(weight, drc_weights[i])

