import os.path

import torch

from spherical_model import SphericalModel
from data_loader import DataLoader
from data_modes import DataModes
from training_loss import TrainingLoss
from data_batch import DataBatch
import cfg


model = SphericalModel().to_device()
optimizer = torch.optim.Adam(model.parameters())
loss_fn = TrainingLoss()
training_loader = DataLoader(DataModes.TRAINING, 0, load_DM=False)

model.train()

for epoch in range(1000) :

    print('epoch %d'%epoch)
    
    for t, data in enumerate(training_loader) :

        print('sample %d / %d'%(t, len(training_loader)))

        assert isinstance(data, DataBatch)
        
        optimizer.zero_grad()

        data = data.to_device()

        prediction = model(data.M200c, data.TNG_radii)

        loss = loss_fn(prediction, data.TNG_Pth)

        loss.backward()

        optimizer.step()

        print('loss = %f'%loss.item())

    torch.save(model.state_dict(), os.path.join(cfg.RESULTS_PATH, 'spherical_model.pt'))
