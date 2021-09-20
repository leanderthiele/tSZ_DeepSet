# tSZ DeepSet

Learn electron pressure in clusters from a gravity-only simulation.
The network architecture consists mostly of DeepSets and MLPs, enabling
a modular, interpretable design that operates directly on the simulation
representation as a set of dark matter (DM) particles.


## Dependencies

* cosmological simulation with gravity-only and full-physics runs from same
  initial conditions.
  We used [IllustrisTNG 300-1](https://www.tng-project.org/data/).
* halo finder. We used [Rockstar](https://bitbucket.org/gfcstanford/rockstar/src/main/).
* [PyTorch](https://pytorch.org/)
* [Optuna](https://optuna.readthedocs.io/en/stable/)
* [Voxelize](https://github.com/leanderthiele/voxelize)
  for [create\_boxes.py](create_boxes.py)
* [group\_particles](https://github.com/leanderthiele/group_particles)
  for [collect\_particles.cpp](collect_particles.cpp)


## Use guide

0. Code that needs to be compiled:
   * [collect\_particles.cpp](collect_particles.cpp): compile into both
     ``collect_particles_DM`` and ``collect_particles_TNG`` executables,
     depending on whether preprocessor macros ``DM`` or ``TNG`` are defined.
     This script finds the simulation particles in the vicinity of halos.
   * [sort\_particles.cpp](sort_particles.cpp) into ``sort_particles``
     executable.
   * [prtfinder.cpp](prtfinder.cpp) into ``libprtfinder.so``.

1. WORKFLOW describes the steps that need to be taken for pre-processing
   of the simulation data.
   The required scripts are:
   * [ascii\_to\_hdf5.py](ascii_to_hdf5.py)
   * [collect\_particles.cpp](collect_particles.cpp)
   * [sort\_particles.py](sort_particles.py)
   * [create\_halo\_catalog.py](create_halo_catalog.py)
   * [create\_boxes.py](create_boxes.py)
   * [create\_Nprt.py](create_Nprt.py)
   * [create\_residuals.py](create_residuals.py)

2. Optional: run [create\_normalization.py](create_normalization.py) and
   [analyse\_normalization.py](analyse_normalization.py), if used
   on a different simulation than IllustrisTNG 300-1. The output can be used
   to tweak the hardcoded numbers in [normalization.py](normalization.py) to
   bring inputs to zero mean, unit variance.

3. Internal data handling is defined in
   * [data\_modes.py](data_modes.py)
   * [data\_loader.py](data_loader.py)
   * [data\_set.py](data_set.py)
   * [data\_batch.py](data_batch.py)
   * [data\_item.py](data_item.py)
   * [halo\_catalog.py](halo_catalog.py)
   * [halo.py](halo.py)

4. The cluster-scale scalars and vectors are defined in
   * [global\_fields.py](global_fields.py)
   * [basis.py](basis.py)
   * [fixed\_len\_vec.py](fixed_len_vec.py)

5. Runtime configuration is defined in [cfg.py](cfg.py).
   The idea is that this module is populated from either command line settings
   or the environment variable ``TSZ_DEEP_SET_CFG`` by [init\_proc.py](init_proc.py).
   For later reference, [archive\_cfg.py](archive_cfg.py) is used.
   Quite often we set default arguments to something from [cfg.py](cfg.py), this
   is implemented in [default\_from\_cfg.py](default_from_cfg.py).

6. There is some basic support for distributed training, although more work would
   be required to get this to work. Currently training is fast enough on a single GPU
   and the required RAM per training process is about 60G.
   Some possibly buggy definitions to set up the distributed environment are in
   [mpi\_env\_types.py](mpi_env_types.py).

7. The network architecture is defined in
   * [network.py](network.py) combined architecture
   * [network\_batt12.py](network_batt12.py) GNFW module
   * [network\_origin.py](network_origin.py) Origin module
   * [network\_local.py](network_local.py) Local module
   * [network\_vae.py](network_vae.py) Stochastic module
   * [network\_decoder.py](network_decoder.py) Aggregator module
   * [network\_deformer.py](network_deformer.py) deviations from spherical symmetry
   * [network\_encoder.py](network_encoder.py) cluster-scale vector DeepSet
   * [network\_scalarencoder.py](network_scalarencoder.py) cluster-scale scalar DeepSet
   * [network\_mlp.py](network_mlp.py) single MLP
   * [network\_layer.py](network_layer.py) DeepSet primitive
   .
   Initialization of the network in [init\_model.py](init_model.py).

8. Driver code is in
   * [training.py](training.py) Can be directly executed with command line arguments
     setting the [cfg.py](cfg.py) settings.
   * [testing.py](testing.py) To test a trained model on either validation or testing set.
   * [optuna\_driver.py](optuna_driver.py) Wrapper around [training.py](training.py) with
     Optuna sampling.

9. The files ``generate_cl_*.py`` are used for Optuna hyperparameter searches.

10. Auxiliary files for training:
    * [training\_loss.py](training_loss.py)
    * [training\_optimizer.py](training_optimizer.py)
    * [training\_loss\_record.py](training_loss_record.py)
    * [plot\_loss.py](plot_loss.py)

11. Auxiliary files for testing:
    * [testing\_loss\_record.py](testing_loss_record.py)
    * [cubify\_prediction.py](cubify_prediction.py)
    * [plot\_testing\_loss.py](plot_testing_loss.py)
    * [plot\_prediction.py](plot_prediction.py)
    * [plot\_profiles.py]((plot_profiles.py)

12. Files ``paper_plot_*.py`` were used to generate publication figures.

13. Various other files are there, most are buggy and not for use. In particular,
    anything with ``FOF`` in the name was for our initial attempt to work with friends-of-friends
    instead of Rockstar halos.
