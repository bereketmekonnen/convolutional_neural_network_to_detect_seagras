import pickle
from matplotlib import  pyplot as plt

dim3_500filname = './global_model_3_2/Single_classifier_del_chl_[]_epochs_500_dim_3/10Oct13_results_10Oct13.p'
dim3_50filname = './global_model_3_2/Single_classifier_del_chl_[]_epochs_50_dim_3/10Oct13_results_10Oct13.p'
dim3_100filname = './global_model_3_2/Single_classifier_del_chl_[]_epochs_100_dim_3/10Oct13_results_10Oct13.p'
dim3_200filname = './global_model_3_2/Single_classifier_del_chl_[]_epochs_200_dim_3/10Oct13_results_10Oct13.p'

dim1_500filname = './global_model_3_2/Single_classifier_del_chl_[]_epochs_500_dim_1/10Oct13_results_10Oct13.p'
dim1_200filname = './global_model_3_2/Single_classifier_del_chl_[]_epochs_200_dim_1/10Oct13_results_10Oct13.p'
#dim1_500filname = './global_model_3_2/Single_classifier_del_chl_[]_epochs_500_dim_1/10Oct13_results_10Oct13.p'

savedirectory = './comparision/'
linewid= 0.8
history3_500 = pickle.load(open(dim3_500filname,'rb'))
history3_50 = pickle.load(open(dim3_50filname,'rb'))
history3_100 = pickle.load(open(dim3_100filname,'rb'))
history3_200 = pickle.load(open(dim3_200filname,'rb'))
history1_200 = pickle.load(open(dim1_200filname,'rb'))

history1_500 = pickle.load(open(dim1_500filname,'rb'))
keys = list(history3_500.history.keys())
plt.clf()
plt.plot(history3_500.history[keys[1]],'y',linewidth=0.8, label = 'dim3_valAcc')
plt.plot(history3_500.history[keys[3]],'r',linewidth=0.8, label = 'dim3_Acc')
plt.plot(history1_500.history[keys[1]],'b',linewidth=0.8, label = 'dim1_valAcc')
plt.plot(history1_500.history[keys[3]],'m',linewidth=0.8, label = 'dim1_Acc' )
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
plt.savefig('dim3_vs_dim1_Acc-ValAcc.png',dpi=600)

plt.clf()
plt.plot(2,history3_500.history[keys[1]][-1],'oy',linewidth=0.8, label = 'dim3_valAcc')
plt.plot(1,history3_500.history[keys[3]][-1],'or',linewidth=0.8, label = 'dim3_Acc')
plt.plot(2,history1_500.history[keys[1]][-1],'ob',linewidth=0.8, label = 'dim1_valAcc')
plt.plot(1,history1_500.history[keys[3]][-1],'om',linewidth=0.8, label = 'dim1_Acc' )
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()


plt.clf()
plt.title("valacc&acc comparison dim3-diff epochs")
#plt.plot(history3_500.history[keys[1]],'y',linewidth=0.8, label = 'epocs500_valAcc')
plt.plot(history3_50.history[keys[1]],'r',linewidth=0.8, label = 'epocs50_valAcc')
plt.plot(history3_100.history[keys[1]],'b',linewidth=0.8, label = 'epocs100_valAcc')
plt.plot(history3_200.history[keys[1]],'m',linewidth=0.8, label = 'epocs200_valAcc')
plt.plot(history3_50.history[keys[3]],'g',linewidth=0.8, label = 'epocs50_Acc')
plt.plot(history3_100.history[keys[3]],'y',linewidth=0.8, label = 'epocs100_Acc')
plt.plot(history3_200.history[keys[3]],'k',linewidth=0.8, label = 'epocs200_Acc')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
plt.savefig('valacc&acc_comparison_dim3_epoc50vs100vs200vs500.png',dpi=600)



plt.clf()
plt.title("valloss&loss comparison dim3-diff epochs")
#plt.plot(history3_500.history[keys[1]],'y',linewidth=0.8, label = 'epocs500_valAcc')
plt.plot(history3_50.history[keys[0]],'r',linewidth=0.8, label = 'epocs50_valLoss')
plt.plot(history3_100.history[keys[0]],'b',linewidth=0.8, label = 'epocs100_valLoss')
plt.plot(history3_200.history[keys[0]],'m',linewidth=0.8, label = 'epocs200_valLoss')
plt.plot(history3_50.history[keys[2]],'g',linewidth=0.8, label = 'epocs50_loss')
plt.plot(history3_100.history[keys[2]],'y',linewidth=0.8, label = 'epocs100_loss')
plt.plot(history3_200.history[keys[2]],'k',linewidth=0.8, label = 'epocs200_loss')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
plt.savefig('valloss&loss_comparison_dim3_epoc50vs100vs200.png',dpi=600)


plt.clf()
plt.title("valLoss&loss comparison dim3vsdim1",fontsize=15)
#plt.plot(history3_500.history[keys[1]],'y',linewidth=0.8, label = 'epocs500_valAcc')

plt.plot(history3_200.history[keys[0]],'m',linewidth=0.8, label = 'Dim3_valLoss')
plt.plot(history3_200.history[keys[2]],'g',linewidth=0.8, label = 'dim3_loss')
plt.plot(history1_200.history[keys[0]],'y',linewidth=0.8, label = 'dim1_valloss')
plt.plot(history1_200.history[keys[2]],'k',linewidth=0.8, label = 'dim1_loss')
plt.ylabel('Loss&valLoss')
plt.xlabel('epoch')
plt.ylim(-0.01, 0.2)
plt.legend()
plt.show()
plt.savefig('valloss&loss_comparison_dim3vsdim1_epoc200.png',dpi=600)


plt.clf()
plt.title("valAcc&Acc comparison dim3vsdim1",fontsize=15)
#plt.plot(history3_500.history[keys[1]],'y',linewidth=0.8, label = 'epocs500_valAcc')

plt.plot(history3_200.history[keys[1]],'m',linewidth=0.8, label = 'Dim3_valAcc')
plt.plot(history3_200.history[keys[3]],'g',linewidth=0.8, label = 'dim3_Acc')
plt.plot(history1_200.history[keys[1]],'y',linewidth=0.8, label = 'dim1_valAcc')
plt.plot(history1_200.history[keys[3]],'k',linewidth=0.8, label = 'dim1_Acc')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.ylim(0.9, 1.01)
plt.legend()
plt.show()
plt.savefig('valAcc&Acc_comparison_ddim3vsdim1_epoc200.png',dpi=300)


plt.clf()
plt.title("val_loss&loss comparison dim3-diff epochs")
#plt.plot(history3_500.history[keys[0]],'y',linewidth=0.8, label = 'epocs500_valAcc')
plt.plot(history3_50.history[keys[0]],'r',linewidth=0.8, label = 'epocs50_valAcc')
plt.plot(history3_100.history[keys[0]],'b',linewidth=0.8, label = 'epocs100_valAcc')
plt.plot(history3_200.history[keys[0]],'m',linewidth=0.8, label = 'epocs200_valAcc')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
plt.savefig('val_loss_dim3_epoc50vs100vs200vs500.png',dpi=600)

##final valacc point
plt.clf()
plt.title("vall acc dim3-diff epochs")
plt.plot(1,history3_500.history[keys[1]][-1],'oy',linewidth=0.8, label = 'epocs500_valAcc')
plt.plot(1,history3_50.history[keys[1]][-1],'or',linewidth=0.8, label = 'epocs50_valAcc')
plt.plot(1,history3_100.history[keys[1]][-1],'ob',linewidth=0.8, label = 'epocs100_valAcc')
plt.plot(1,history3_200.history[keys[1]][-1],'om',linewidth=0.8, label = 'epocs200_valAcc')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
plt.savefig('dim3_epoc50vs100vs200vs500_final_valacc.png',dpi=600)