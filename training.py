import torch
import torch.nn as nn

from mpi_env_types import MPIEnvTypes
from network import Network
from data_loader import DataLoader
from data_modes import DataModes
from training_loss import TrainingLoss
from data_batch import DataBatch
from init_proc import InitProc
import cfg

EPOCHS = 500

InitProc(0)

if cfg.MPI_ENV_TYPE is MPIEnvTypes.NOGPU :
    torch.set_num_threads(5)

model = Network().to_device()
optimizer = torch.optim.Adam(model.parameters())
loss_fn = TrainingLoss()

training_loader = DataLoader(mode=DataModes.TRAINING)

scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2,
                                                steps_per_epoch=len(training_loader),
                                                epochs=EPOCHS)


for epoch in range(EPOCHS) :

    model.train()

    for t, data in enumerate(training_loader) :

        optimizer.zero_grad()
        data = data.to_device()
        prediction = model(data)
        loss = loss_fn(prediction, data.TNG_Pth)
        loss.backward()
        
        if cfg.GRADIENT_CLIP is not None :
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRADIENT_CLIP)

        optimizer.step()
        scheduler.step()
