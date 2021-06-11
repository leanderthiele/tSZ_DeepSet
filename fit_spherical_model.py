import os.path

# FIXME
if False :
    import numpy as np

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

        # FIXME for debugging
        if False :
            np.savez('test.npz',
                     pred=np.squeeze(prediction.detach().numpy()),
                     targ=np.squeeze(data.TNG_Pth.detach().numpy()),
                     r=np.squeeze(data.TNG_radii.detach().numpy()))
            raise RuntimeError

        loss = loss_fn(prediction, data.TNG_Pth)

        loss.backward()

        if cfg.GRADIENT_CLIP is not None :
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRADIENT_CLIP)

        optimizer.step()

        print('loss = %f'%loss.item())

    torch.save(model.state_dict(), os.path.join(cfg.RESULTS_PATH, 'spherical_model.pt'))
