import torch
import torch.nn as nn

from data_modes import DataModes
from data_loader import DataLoader
from training_loss import TrainingLoss
from data_batch import DataBatch
from origin import Origin
from network_origin import NetworkOrigin
from global_fields import GlobalFields
from basis import Basis
import cfg

# FIXME for debugging on head node
torch.set_num_threads(4)

model = NetworkOrigin().to_device()
optimizer = torch.optim.Adam(model.parameters())
loss_fn = TrainingLoss()
training_loader = DataLoader(DataModes.TRAINING, 0, load_TNG=False, origin=Origin.CM)

model.train()

for epoch in range(1000) :

    print('epoch %d'%epoch)

    for t, data in enumerate(training_loader) :
    
        print('sample %d / %d'%(t, len(training_loader)))

        assert isinstance(data, DataBatch)

        optimizer.zero_grad()

        data = data.to_device()

        prediction = model(data.DM_coords,
                           u=data.u if len(GlobalFields) != 0 else None,
                           basis=data.basis if len(Basis) != 0 else None)

        cm = data.CM_DM
        if cfg.NORMALIZE_COORDS :
            cm /= data.R200c.unsqueeze(-1)

        prediction += cm

        target = data.pos_TNG
        if cfg.NORMALIZE_COORDS :
            target /= data.R200c.unsqueeze(-1)

        loss = loss_fn(prediction, target.unsqueeze(1))

        loss.backward()

        if cfg.GRADIENT_CLIP is not None :
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRADIENT_CLIP)

        optimizer.step()

        print('loss = %f'%loss.item())
