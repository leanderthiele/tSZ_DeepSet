"""
Small script to get the inputs to our networks normalized to
standard normal.

Uses the create_scalars methods of various networks to see directly
what they would look like.
"""

from init_proc import InitProc
import cfg

InitProc(0)

cfg.NET_ARCH = dict(origin=True,
                    batt12=True,
                    deformer=True,
                    encoder=True,
                    decoder=True,
                    vae=True,
                    local=True)
