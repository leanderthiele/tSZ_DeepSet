import os.path

import torch
import torch.nn as nn

from data_modes import DataModes
from data_loader import DataLoader
from training_loss import TrainingLoss
from data_batch import DataBatch
from network_origin import NetworkOrigin
from global_fields import GlobalFields
from basis import Basis
from init_proc import InitProc
import cfg

# FIXME for debugging on head node
# torch.set_num_threads(4)

InitProc(0)
print(cfg.MPI_ENV_TYPE)
print(cfg.DEVICE_IDX)

model = NetworkOrigin().to_device()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = TrainingLoss()
training_loader = DataLoader(mode=DataModes.TRAINING, load_TNG=False)
validation_loader = DataLoader(mode=DataModes.VALIDATION, load_TNG=False)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2,
                                                steps_per_epoch=len(training_loader),
                                                epochs=1000)


for epoch in range(1000) :

    print('epoch %d'%epoch)

    print('TRAINING')
    model.train()

    for t, data in enumerate(training_loader) :

        assert isinstance(data, DataBatch)

        optimizer.zero_grad()

        data = data.to_device()

        prediction = model(data.DM_coords,
                           u=data.u if len(GlobalFields) != 0 else None,
                           basis=data.basis if len(Basis) != 0 else None)

        # remove the singleton feature channel dimension
        prediction = prediction.squeeze(dim=1)

        cm = data.CM_DM
        if cfg.NORMALIZE_COORDS :
            cm /= data.R200c.unsqueeze(-1)

        prediction += cm

        target = data.pos_TNG
        if cfg.NORMALIZE_COORDS :
            target /= data.R200c.unsqueeze(-1)

        loss = loss_fn(prediction, target)

        loss.backward()

        if cfg.GRADIENT_CLIP is not None :
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRADIENT_CLIP)

        optimizer.step()

        scheduler.step()

        training_loss = loss.item()
        guess_loss = loss_fn(cm, target).item()
        print('training loss = %f vs guess = %f %s %s'%(training_loss, guess_loss, '***' if guess_loss>0.1 else '', '+++' if training_loss>0.1 else ''))

    torch.save(model.state_dict(), os.path.join(cfg.RESULTS_PATH, 'origin.pt'))

    print('VALIDATION')
    model.eval()

    for t, data in enumerate(validation_loader) :
        
        data = data.to_device()

        prediction = model(data.DM_coords,
                           u=data.u if len(GlobalFields) != 0 else None,
                           basis=data.basis if len(Basis) != 0 else None)

        prediction = prediction.squeeze(dim=1)

        cm = data.CM_DM
        if cfg.NORMALIZE_COORDS :
            cm /= data.R200c.unsqueeze(-1)

        prediction += cm

        target = data.pos_TNG
        if cfg.NORMALIZE_COORDS :
            target /= data.R200c.unsqueeze(-1)

        loss = loss_fn(prediction, target)

        validation_loss = loss.item()
        guess_loss = loss_fn(cm, target).item()
        print('validation loss = %f vs guess = %f %s %s'%(validation_loss, guess_loss, '***' if guess_loss>0.1 else '', '+++' if validation_loss>0.1 else ''))
