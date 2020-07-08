import torch
from torchvision.utils import make_grid
import torch.nn.functional as f
from SPC_2L.DataTools import norm
import pickle

#import matplotlib.pyplot as plt
import numpy as np
import gc

class Monitor(object):
    def __init__(self, Net, writer, n_row=None, DicoL=True):
        self.writer = writer
        self.Net = Net
        self.idx = 0
        self.n_row = n_row
        if self.n_row is None:
            self.n_row = [16]*self.Net.nb_layers
        #self.act = [torch.zeros()]*self.Net.nb_layers
        if torch.cuda.is_available():
            self.total_activity = [torch.zeros(1,1,1,1).cuda()]*self.Net.nb_layers
            self.act = [torch.zeros(1,self.Net.layers[i].dico_shape[0],1,1).cuda() for i in range(self.Net.nb_layers)]
        else :
            self.total_activity = [torch.zeros(1, 1, 1, 1)] * self.Net.nb_layers
            self.act = [torch.zeros(1, self.Net.layers[i].dico_shape[0], 1, 1) for i in range(self.Net.nb_layers)]
       # self.fig = [plt.subplots()] * self.Net.nb_layers
       # self.x_bar = [np.arange(self.Net.layers[i].dico.data.size()[0]) + 1 for i in range(self.Net.nb_layers)]

        if DicoL:
            self.all_stride_list = [self.Net.layers[0].stride]
            for i in range(Net.nb_layers - 1):
                if self.Net.layers[i].transform is not None:
                    stride = self.Net.layers[i].transform.stride
                else:
                    stride = 1
                self.all_stride_list.append(self.all_stride_list[-1] * stride * self.Net.layers[i+1].stride)
            if torch.cuda.is_available():
                self.size_eff = [self.Net.project_dico(i,cpu=False).size() for i in range(self.Net.nb_layers)]
                self.D_eff = [torch.randn(self.size_eff[i]).cuda() for i in range(self.Net.nb_layers)]
                self.D_eff_m = [torch.randn(self.size_eff[i]).cuda() for i in range(self.Net.nb_layers)]
            else:
                self.size_eff = [self.Net.project_dico(i, cpu=True).size() for i in range(self.Net.nb_layers)]
                self.D_eff = [torch.randn(self.size_eff[i]) for i in range(self.Net.nb_layers)]
                self.D_eff_m = [torch.randn(self.size_eff[i]) for i in range(self.Net.nb_layers)]


    def MonitorLoss(self, loss, k, separate=False):
        for idx_layer in range(self.Net.nb_layers):
            if separate:
                self.writer.add_scalar('5a_LL/L{0}'.format(idx_layer), loss.value[idx_layer][0], k)
                self.writer.add_scalar('5b_UL/L{0}'.format(idx_layer), loss.value[idx_layer][1], k)
            else:
                self.writer.add_scalar('5_Loss/L{0}'.format(idx_layer),
                                       loss.value[idx_layer][0] + loss.value[idx_layer][1], k)

    def MonitorGamma(self, gamma, k, option=['NNZ', '%', 'Mean', 'Sum', 'V']):
        for i in range(self.Net.nb_layers):
            active_els = (gamma[i] != 0).float().sum().div(self.Net.sparse_map_size[i][0])
            if 'NNZ' in option:
                self.writer.add_scalar('1_Sparsity_NNZ/L{}'.format(i), active_els, k)

            volume = float((gamma[i].size()[-1] * gamma[i].size()[-2] * gamma[i].size()[-3]))
            relative_sparsity = 100 - (active_els.float() / volume) * 100

            if '%' in option:
                self.writer.add_scalar('0_Sparsity_Pct/L{}'.format(i), relative_sparsity, k)

            if 'Mean' in option:
                avg = torch.mean(self.Net.layers[i].v * gamma[i][gamma[i] != 0])
                self.writer.add_scalar('2_Mean_Activity/L{0}'.format(i), avg, k)

            if 'Sum' in option:
                activity = gamma[i].sum().div(self.Net.sparse_map_size[i][0])
                self.writer.add_scalar('4_Sum/L{0}:'.format(i), activity, k)

            if 'V' in option:
                if self.Net.layers[i].v_size is not None:
                    self.writer.add_scalar('3_V/L{}'.format(i), self.Net.layers[i].v.detach().mean(), k)

    def MonitorDicoBP(self, k):
        for i in range(self.Net.nb_layers):
            name = 'DicoBP/L_{0}'.format(i+1)
            dico_img = make_grid(self.Net.project_dico(i), normalize=True, pad_value=1, nrow=self.n_row[i])
            self.writer.add_image(name, dico_img, k)


    def MonitorDicoL(self, batch, gamma, k, nb_iter=50, l_r=1e-3, m=0.9):
        for i in range(self.Net.nb_layers):
        #for i in range(self.Net.nb_layers-1, -1, - 1):

            nb_iter=1
            for idx in range(nb_iter):
                self.D_eff[i].requires_grad = True
                loss_D_eff = (batch - f.conv_transpose2d(gamma[i], self.D_eff[i], stride=self.all_stride_list[i],
                                                         padding=self.Net.layers[i].pad*self.all_stride_list[i],
                                                         output_padding=self.Net.layers[i].out_pad)).pow(2).sum().div(2)
                #loss_D_eff = (batch - f.conv_transpose2d(gamma[i], self.D_eff[i], stride=self.all_stride_list[i],
                #                                         output_padding=self.Net.layers[i].out_pad)).pow(2).sum().div(2)
                loss_D_eff /= batch.size()[0]
                loss_D_eff.backward()
                self.D_eff[i].requires_grad = False
                self.D_eff_m[i] = m * self.D_eff_m[i] - self.D_eff[i].grad
                self.D_eff[i] += l_r * self.D_eff_m[i]
                self.D_eff[i].grad.zero_()
                self.D_eff[i] /= norm(self.D_eff[i])

            dico_img = make_grid(self.D_eff[i], normalize=True, pad_value=1, nrow=self.n_row[i])
            self.writer.add_image('DicoL/L{}'.format(i), dico_img, k)


    def MonitorList(self, list, name, k):
        #if len(list) != self.Net.nb_layers:
        #    print('Error ! The size of the list to monitor should be the number of layer')
        for i in range(self.Net.nb_layers):
            self.writer.add_scalar(name + '/L{0}:'.format(i), list[i], k)


    def MonitorBatch(self):
        print('need to be done')

    def ComputeHisto(self,gamma):

        for i in range(self.Net.nb_layers):
            self.total_activity[i] += gamma[i].abs().sum().add(1e-8)
            self.act[i] += gamma[i].abs().sum(0, keepdim=True).sum(-1, keepdim=True).sum(-2, keepdim=True)


    def Save(self, file_name):
        to_save={'dico_eff': [self.D_eff[i].cpu() for i in range(self.Net.nb_layers)]}

        with open(file_name, 'wb') as file:
            pickle.dump(to_save, file, pickle.HIGHEST_PROTOCOL)
        print('File Saved at location : {0}'.format(file_name))

    '''
    def PlotHisto(self, k):

        for i in range(self.Net.nb_layers):
            activity = (self.act[i]/self.total_activity[i])
            activity = activity[0,:,0,0].cpu().numpy()

            #self.fig[i][1].bar(self.x_bar[i], self.act[i], width=1.5)
            #self.fig[i][1].bar(np.arange(self.Net.layers[i].dico.data.size()[0]) + 1, self.act[i], width=1.5)
            #self.fig[i][1].set_xlabel('Dico Number')
            #self.fig[i][1].set_ylabel('Fraction of Activity')

            #print(self.fig[0])
            #stop
            fig, ax = plt.subplots()

            #ax = self.fig.add_subplot(1, 1, 1)
            ax.bar(np.arange(self.Net.layers[i].dico.data.size()[0]) + 1, activity, width=1.5)
            ax.set_xlabel('Dico Number')
            ax.set_ylabel('Fraction of Activity')




            name = 'Activity_Bar/L{}'.format(i + 1)
            self.writer.add_figure(name, fig, k)
            ## Release memory
            fig.clf()
            plt.close('all')
            del fig, ax
            gc.collect()
    '''
