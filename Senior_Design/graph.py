import pickle
from matplotlib import pyplot as plt
import glob
import plotly.graph_objects as go
import os
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from scipy.interpolate import make_interp_spline
keys = ['val_loss', 'val_accuracy', 'loss', 'accuracy']

# file names
dim3_500_ss5kfilname = './global_model_3_2/Single_classifier_del_chl_[]_epochs_500_dim_3_spc_5000_batchS_256/10Oct13_results_10Oct13.p'
dim3_500_ss10kfilname = './global_model_3_2/Single_classifier_del_chl_[]_epochs_500_dim_3_spc_10000_batchS_256/10Oct13_results_10Oct13.p'
dim3_500_ss15kfilname = './global_model_3_2/Single_classifier_del_chl_[]_epochs_500_dim_1_spc_15000_batchS_256/10Oct13_results_10Oct13.p'
dim1_500_ss15kfilname = './global_model_3_2/Single_classifier_del_chl_[]_epochs_500_dim_3_spc_15000_batchS_256/10Oct13_results_10Oct13.p'
dim3_500_ss20kfilname = './global_model_3_2/Single_classifier_del_chl_[]_epochs_500_dim_3_spc_20000_batchS_256/10Oct13_results_10Oct13.p'

# loading the history of neural network
hist_500e_5000ss = pickle.load(open(dim3_500_ss5kfilname, 'rb'))
hist_500e_10000ss = pickle.load(open(dim3_500_ss10kfilname, 'rb'))
hist3_500e_15000ss = pickle.load(open(dim3_500_ss15kfilname, 'rb'))
hist1_500e_15000ss = pickle.load(open(dim1_500_ss15kfilname, 'rb'))
hist_500e_20000ss = pickle.load(open(dim3_500_ss20kfilname, 'rb'))

plt.clf()
plt.plot(hist3_500e_15000ss.history[keys[0]], linewidth=0.8, label='dim3_valloss')
plt.plot(hist3_500e_15000ss.history[keys[1]], linewidth=0.8, label='dim3_valacc')
plt.plot(hist1_500e_15000ss.history[keys[0]], linewidth=0.8, label='dim1_valloss')
plt.plot(hist1_500e_15000ss.history[keys[1]], linewidth=0.8, label='dim1_vallacc')
plt.ylabel('val_accuracy,val loss')
plt.xlabel('epoch')
plt.title('DIM 3 VS DIM 1 testing results')
plt.legend()
plt.show()
plt.savefig('./Graphs/DIM 3 VS DIM 1 testing results.png', dpi=600)

plt.clf()
plt.plot(hist3_500e_15000ss.history[keys[2]], 'b', linewidth=0.8, label='dim3_loss')
plt.plot(hist3_500e_15000ss.history[keys[3]], 'c', linewidth=0.8, label='dim3_acc')
plt.plot(hist1_500e_15000ss.history[keys[2]], 'r', linewidth=0.8, label='dim1_loss')
plt.plot(hist1_500e_15000ss.history[keys[3]], 'y', linewidth=0.8, label='dim1_acc')
plt.ylabel('val_accuracy,val loss')
plt.xlabel('epoch')
plt.title('DIM 3 VS DIM 1 traning results')
plt.legend()
plt.show()
plt.savefig('./Graphs/DIM 3 VS DIM 1 traning results.png', dpi=600)


f = (glob.glob('./global_model_3_2_modified/' + "*/10Oct13_results_10Oct13.p"))
f = (glob.glob('./global_model_3_2_modified/' + "*"))
f.sort(key=os.path.getctime)
for i, a in enumerate(f):
    print(f'i:{i}, a:{a}')




del_chanl_5     = pickle.load(open(f[10], 'rb'))
del_chanl_6     = pickle.load(open(f[11], 'rb'))
del_chanl_5_6   = pickle.load(open(f[2], 'rb'))

# template :
# plt.clf()
# plt.plot(___.history[keys[1]]  , linewidth=0.8, label='del_')
# plt.title("Validation Accuracy")
# plt.ylabel('Val Accuracy')
# plt.xlabel('epoch')
# plt.legend()
# plt.show()
# plt.savefig('./Graphs_modified/dim_E_SS_15k_.png', dpi=600)

plt.ioff()
# del_chanl_4_5_6 = pickle.load(open(f[0],'rb'))
plt.clf()
plt.plot(del_chanl_5.history[keys[1]]  , linewidth=0.8, label='del_5  ')
plt.plot(del_chanl_6.history[keys[1]]  , linewidth=0.8, label='del_6  ')
plt.plot(del_chanl_5_6.history[keys[1]], linewidth=0.8, label='del_5_6')
plt.title("Validation Accuracy")
plt.ylabel('Val Accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
plt.savefig('./Graphs_modified/dim3_200E_SS_15k_dif_5v6chanels_wt4_valacc.png', dpi=600)

plt.clf()
plt.plot(del_chanl_5.history[keys[0]]  , linewidth=0.8, label='del_5  ')
plt.plot(del_chanl_6.history[keys[0]]  , linewidth=0.8, label='del_6  ')
plt.plot(del_chanl_5_6.history[keys[0]], linewidth=0.8, label='del_5_6')
plt.title("Validation Loss")
plt.ylabel('Val Loss')
plt.xlabel('epoch')
plt.legend()
plt.show()
plt.savefig('./Graphs_modified/dim3_200E_SS_15k_dif_5v6chanels_wt4_valloss.png', dpi=600)

plt.clf()
plt.plot(del_chanl_0_5.history[keys[0]], linewidth=0.8, label='del_0_5')
plt.plot(del_chanl_0_6.history[keys[0]], linewidth=0.8, label='del_0_6')
plt.plot(del_chanl_5_6.history[keys[0]], linewidth=0.8, label='del_5_6')
plt.title("Validation Loss")
plt.ylabel('Val loss')
plt.xlabel('epoch')
plt.legend()
# plt.show()
plt.savefig('./Graphs_modified/dim3_200E_SS_15k_dif_5chanels_wt4_valoss.png', dpi=600)

# loading the history of neural network


# fig = go.Figure(data=[go.Table(header=dict(values=['neural net', 'Acc']),
#                  cells=dict(values=[['SS5k', 'SS10k', 'SS15k', 'SS20k'],
#                                     [hist_500e_5000ss.history[keys[3]][-1],
#                                      hist_500e_10000ss.history[keys[3]][-1],
#                                      hist_500e_15000ss.history[keys[3]][-1],
#                                      hist_500e_20000ss.history[keys[3]][-1]]]))
#                      ])
# fig.update_layout(width=500, height=300)
# fig.show()
# fig.write_image('./Graphs/dim3_500E_SS_5k_vs_10k_vs_15k_vs_20k_acc_table.png')


dim3_500_ss15k_filname = './global_model_3_2/Single_classifier_del_chl_[]_epochs_500_dim_3_spc_15000_batchS_256/10Oct13_results_10Oct13.p'
dim3_500_ss15k_modfilname = './global_model_3_2_modified/Single_classifier_del_chl_[]_epochs_500_dim_3_spc_15000_batchS_256/10Oct13_results_10Oct13.p'

dim3_500_ss15k = pickle.load(open(dim3_500_ss15k_filname, 'rb'))
dim3_500_ss15k_mod = pickle.load(open(dim3_500_ss15k_modfilname, 'rb'))

plt.clf()
plt.plot(dim3_500_ss15k.history[keys[0]], 'b', linewidth=0.8, label='val_loss_org')
plt.plot(dim3_500_ss15k_mod.history[keys[0]], 'g', linewidth=0.8, label='val_loss_mod')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()
plt.savefig('./Graphs/dim3_500E_SS_15k_ org_vs_mod_valloss.png', dpi=600)

plt.clf()
plt.plot(dim3_500_ss15k.history[keys[1]], 'b', linewidth=0.8, label='val_acc_org')
plt.plot(dim3_500_ss15k_mod.history[keys[1]], 'g', linewidth=0.8, label='val_acc_mod')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()
plt.savefig('./Graphs/dim3_500E_SS_15k_ org_vs_mod_valacc.png', dpi=600)
print(f"org vs mod final valacc:{dim3_500_ss15k.history[keys[1]][-1]},{dim3_500_ss15k_mod.history[keys[1]][-1]}")
print(f"org vs mod final valloss:{dim3_500_ss15k.history[keys[0]][-1]},{dim3_500_ss15k_mod.history[keys[0]][-1]}")

