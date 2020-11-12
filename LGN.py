import os, sys
import pdb
import numpy as np
import yaml

import torch
import torch.nn as nn
import pandas as pd

sys.path.append('../bmtk')
from bmtk.simulator.filternet.lgnmodel.temporalfilter import TemporalFilterCosineBump
from bmtk.simulator.filternet.lgnmodel.spatialfilter import GaussianSpatialFilter
from bmtk.simulator.filternet.lgnmodel.linearfilter import SpatioTemporalFilter


class Conv3dLGN(nn.Conv3d):
    def __init__(self,in_channels,out_channels,kernel_size):
        super(Conv3dLGN,self).__init__(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size, padding=kernel_size//2)
        self._set_weights()   

    def forward(self):
        pass
    
    def _set_weights(self):
        cell_specs = yaml.load(open('./cell_specs.yaml'), Loader=yaml.FullLoader)
        cell_types = cell_specs['cell_types']
        num_cells_per_type = cell_specs['num_cells']
        i =0 
        for cell_type, num_cells in zip(cell_types,num_cells_per_type):
            kernels = self._make_kernels(cell_type, num_cells)    
            self.weight[i:(i+num_cells),:,:,:,:] = nn.Parameter(kernels, requires_grad=False)
            i += num_cells

    def _load_param_values(self):

        param_file_path = './lgn_full_col_cells_3.csv'
        lgn_types_file_path = './lgn_full_col_cell_models_3.csv'

        param_table = pd.read_csv(param_file_path, sep=' ')
        lgn_types_table = pd.read_csv(lgn_types_file_path)

        return param_table, lgn_types_table

    def _make_kernels(self,cell_type, num_cells):

        param_table, lgn_types_table = self._load_param_values()

        all_spatial_sizes = param_table['spatial_size'][param_table['model_id']==cell_type]
        all_kpeaks_dom_0s = param_table['kpeaks_dom_0'][param_table['model_id']==cell_type]
        all_kpeaks_dom_1s = param_table['kpeaks_dom_1'][param_table['model_id']==cell_type]
        all_weight_dom_0s = param_table['weight_dom_0'][param_table['model_id']==cell_type]
        all_weight_dom_1s = param_table['weight_dom_1'][param_table['model_id']==cell_type]
        all_delay_dom_0s = param_table['delay_dom_0'][param_table['model_id']==cell_type]
        all_delay_dom_1s = param_table['delay_dom_1'][param_table['model_id']==cell_type]
        if cell_type == 'sONsOFF' or cell_type == 'sONtOFF':
            all_kpeaks_non_dom_0s = param_table['kpeaks_non_dom_0'][param_table['model_id']==cell_type]
            all_kpeaks_non_dom_1s = param_table['kpeaks_non_dom_1'][param_table['model_id']==cell_type]
            all_weight_non_dom_0s = param_table['weight_non_dom_0'][param_table['model_id']==cell_type]
            all_weight_non_dom_1s = param_table['weight_non_dom_1'][param_table['model_id']==cell_type]
            all_delay_non_dom_0s = param_table['delay_non_dom_0'][param_table['model_id']==cell_type]
            all_delay_non_dom_1s = param_table['delay_non_dom_1'][param_table['model_id']==cell_type]

        if 'sOFF' in cell_type:
            amplitude = -1.0
        else:
            amplitude = 1.0
        kdom_data = torch.empty((num_cells,3,*self.kernel_size))
        knondom_data = torch.empty((num_cells,3,*self.kernel_size))
        for cellcount in range(0,num_cells):
            
            sampled_cell_idx = int(torch.randint(low=min(all_kpeaks_dom_0s.keys()),high=max(all_kpeaks_dom_0s.keys()),size=(1,1)))
            # print(sampled_cell_idx)

            Tdom = TemporalFilterCosineBump(weights=(all_weight_dom_0s[sampled_cell_idx],all_weight_dom_1s[sampled_cell_idx]), 
                                            kpeaks=(all_kpeaks_dom_0s[sampled_cell_idx],all_kpeaks_dom_1s[sampled_cell_idx]), 
                                            delays=(all_delay_dom_0s[sampled_cell_idx],all_delay_dom_1s[sampled_cell_idx]))
            

            this_sigma = all_spatial_sizes[sampled_cell_idx]
            Sdom = GaussianSpatialFilter(translate=(0.0, 0.0), 
                                        sigma=(this_sigma, this_sigma), 
                                        rotation=0, 
                                        origin='center')

            Kerneldom = SpatioTemporalFilter(spatial_filter = Sdom, temporal_filter = Tdom, amplitude=amplitude)
            # Kerneldom.show_temporal_filter(show=True)
            # Kerneldom.show_spatial_filter(row_range=range(0,10),col_range=range(0,10),show=True)
            kdom = Kerneldom.get_spatiotemporal_kernel(row_range=range(0,10),col_range=range(0,10))
            kdom_data[cellcount,:,:,:,:] = torch.Tensor(kdom.full())[::60,:,:].repeat([3,1,1,1])

            if cell_type == 'sONsOFF' or cell_type == 'sONtOFF':
                Tnondom = TemporalFilterCosineBump(weights=(all_weight_non_dom_0s[sampled_cell_idx],all_weight_non_dom_1s[sampled_cell_idx]), 
                                            kpeaks=(all_kpeaks_non_dom_0s[sampled_cell_idx],all_kpeaks_non_dom_1s[sampled_cell_idx]), 
                                            delays=(all_delay_non_dom_0s[sampled_cell_idx],all_delay_non_dom_1s[sampled_cell_idx]))

            
                Snondom = GaussianSpatialFilter(translate=(0.0, 0.0), 
                                            sigma=(this_sigma, this_sigma), 
                                            rotation=0, 
                                            origin='center')
                
                Kernelnondom = SpatioTemporalFilter(spatial_filter = Snondom, temporal_filter = Tnondom, amplitude=amplitude)
                knondom = Kerneldom.get_spatiotemporal_kernel(row_range=range(0,10),col_range=range(0,10))
                knondom_data[cellcount,:,:,:,:] = torch.Tensor(knondom.full())[::60,:,:].repeat([3,1,1,1])
            
            return kdom_data#, knondom_data






if  __name__ == "__main__":
    import matplotlib.pyplot as plt

    lgn = Conv3dLGN(in_channels = 3,out_channels = 12, kernel_size = 10)
    # kdom_data = lgn._set_weights(cell_type='sON_TF8',num_cells=1)
    # p,f = lgn._load_param_values()
    # plt.plot(kdom_data[::60,0,0]), plt.show()
    # plt.imshow(kdom_data[0,:,:]), plt.show()

    pdb.set_trace()
