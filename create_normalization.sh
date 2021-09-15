python -u create_normalization.py \
  --'NET_ARCH=dict(origin=True, batt12=True, deformer=True, encoder=True, decoder=True, vae=True, local=True)' \
  --'PRT_FRACTION["DM"]["training"]=1e3' \
  --'PRT_FRACTION["TNG"]["training"]=256' \
  --'GLOBALS_USE["logM"]=True' \
  --'GLOBALS_USE["Xoff"]=True' \
  --'GLOBALS_USE["Voff"]=True' \
  --'GLOBALS_USE["CM"]=True' \
  --'GLOBALS_USE["ang_mom"]=True' \
  --'GLOBALS_USE["inertia"]=True' \
  --'GLOBALS_USE["inertia_dot_ang_mom"]=True' \
  --'GLOBALS_USE["vel_dispersion"]=True' \
  --'GLOBALS_USE["vel_dispersion_dot_ang_mom"]=True' \
  --'GLOBALS_USE["vel_dispersion_dot_inertia"]=True' \
  --'BASIS_USE["ang_mom"]=True' \
  --'BASIS_USE["CM"]=True' \
  --'BASIS_USE["inertia"]=True' \
  --'BASIS_USE["vel_dispersion"]=True' \
  --'DATALOADER_ARGS["num_workers"]=1' \
  --'LOCAL_PASS_N=True' \
  --'R_LOCAL=300.0'
