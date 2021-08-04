


source pytorch_env.sh

python -u create_normalization.py \
  --'NET_ARCH=dict(origin=True, batt12=True, deformer=True, encoder=True, decoder=True, vae=True, local=True)' \
  --'PRT_FRACTION["TNG"]=256' \
  --'GLOBALS_USE["none"]=False' \
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
  --'BASIS_USE["none"]=False' \
  --'BASIS_USE["ang_mom"]=True' \
  --'BASIS_USE["CM"]=True' \
  --'BASIS_USE["inertia"]=True' \
  --'BASIS_USE["vel_dispersion"]=True'
