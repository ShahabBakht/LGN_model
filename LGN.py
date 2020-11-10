import torch
import torch.nn as nn


class convLGN(nn.Module):
    def __init__(self):
        super(convLGN).__ini
        pass

    def forward(self):
        pass

    def set_weights(self):
        pass

if  __name__ == "__main__":
    import os, sys
    import pdb
    sys.path.append('../bmtk')
    from bmtk.simulator.filternet.lgnmodel.temporalfilter import TemporalFilterCosineBump
    from bmtk.simulator.filternet.lgnmodel.spatialfilter import GaussianSpatialFilter
    from bmtk.simulator.filternet.lgnmodel.linearfilter import SpatioTemporalFilter

    T = TemporalFilterCosineBump([3,-1], [13,27], [0,25])
    S = GaussianSpatialFilter(translate=(0.0, 0.0), sigma=(1.0, 1.0), rotation=0, origin='center')

    Kernel = SpatioTemporalFilter(spatial_filter = S, temporal_filter = T, amplitude=1.)

    pdb.set_trace()
