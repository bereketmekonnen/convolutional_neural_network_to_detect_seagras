from datetime import datetime

start = datetime.now()
from SJB_modified_funtion_classifier import run_cnn
from tqdm import tqdm
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
import pickle
from matplotlib import pyplot as plt
import numpy as np

plt.ioff()

dimension = 3
deleted_channels = [[5, 6],[4],[4,6,5]]
# selected_sample_per_class=[x*1000 for x in [7,10,15,20]]
selected_sample_per_class = 10000
balanced_option = 'balanced'
epochs = 200
batchSize = 256
# batchSize = [2**x for x in range(8,10)]
numOfClasses = 5
testPercent = 0.1

dif = deleted_channels
difname = "CD"  # E: epochs, CD: chanels deleted, BS : batchsize, SCP: seleted samples.. , D : dimention
i = 1
ttl = len(dif)


for d in dif:
    print(f'================\n\t LOOP {i}/{ttl}\n================')
    i += 1
    run_cnn(dimension=d if dif == dimension else dimension,
            deleted_channels=d if dif == deleted_channels else deleted_channels,
            selected_sample_per_class=d if dif == selected_sample_per_class else selected_sample_per_class,
            balanced_option=d if dif == balanced_option else balanced_option,
            epochs=d if dif == epochs else epochs,
            batchSize=d if dif == batchSize else batchSize,
            numOfClasses=numOfClasses,
            testPercent=d if dif == testPercent else testPercent)
    print("\n\n========================\n\n========================\n\n")
# for deleted_channels in deleted_channels:
#     print(f'================\n\t LOOP {i}/{ttl}\n================')
#     i += 1
#     run_cnn(dimension = dimension,
#         deleted_channels= deleted_channels,
#         selected_sample_per_class= selected_sample_per_class,
#         balanced_option=balanced_option,
#         epochs=epochs,
#         batchSize=batchSize,
#         numOfClasses = numOfClasses ,
#         testPercent =testPercent)
#     print("\n\n========================\n\n========================\n\n")


# Creating Graphs :

depth = lambda L: isinstance(L, list) and max(map(depth, L)) + 1


def vs(l):
    if (type(l) == type([])):
        return "_VS_".join(map(str, l))
    return str(l)


def vs_d(l):
    if len(l) > 0 and depth(l) > 1:
        return "_VS_".join(map(str, l))
    return str(l)


def createGraphTitle():
    title = 'constants: ' + \
            str(f'CD : {deleted_channels},' if dif != deleted_channels else "") + \
            str(f'E : {epochs}, ' if dif != epochs else "") + \
            str(f'Dim : {dimension}, ' if dif != dimension else "") + \
            str(f'SCP : {selected_sample_per_class}, ' if dif != selected_sample_per_class else "") + \
            str(f'BS : {batchSize}, ' if dif != batchSize else '')
    return title


directorys = []
hs = []
for d in dif:
    directory = './global_model_3_2_modified/' + \
                'Single_classifier_del_chl_' + str(d if dif == deleted_channels else deleted_channels) + \
                '_epochs_' + str(d if dif == epochs else epochs) + \
                '_dim_' + str(d if dif == dimension else dimension) + \
                '_spc_' + str(d if dif == selected_sample_per_class else selected_sample_per_class) + \
                '_batchS_' + str(d if dif == batchSize else batchSize) + '/'
    print(directory)
    directorys.append(directory)
    del directory
    del d
# loading history of models
for directory in directorys:
    hs.append(pickle.load(open(directory + '10Oct13_results_10Oct13.p', 'rb')))

gname = 'Single_classifier_del_chl_' + vs_d(deleted_channels) + \
        '_epochs_' + vs(epochs) + \
        '_dim_' + vs(dimension) + \
        '_spc_' + vs(selected_sample_per_class) + \
        '_batchS_' + vs(batchSize) + '.png'
keys = list(hs[0].history.keys())

# graphing Val ACC
plt.clf()
for i, h in enumerate(hs):
    plt.plot(h.history[keys[1]], linewidth=0.8, label=f'{difname}_{dif[i]}')
plt.suptitle(createGraphTitle())
plt.title("Validation Accuracy")
plt.ylabel('Val Accuracy')
plt.xlabel('epoch')
# axes = plt.gca()
# axes.set_ylim([0.96,1])
plt.legend()
# plt.show()
plt.savefig(f'./Graphs_modified/{gname}_Valacc_.png', dpi=600)

# Graphing Val Loss
plt.clf()
for i, h in enumerate(hs):
    plt.plot(h.history[keys[0]], linewidth=0.8, label=f'{difname}_{dif[i]}')
plt.suptitle(createGraphTitle())
plt.title("Validation Loss")
plt.ylabel('Val Loss')
plt.xlabel('epoch')
# axes = plt.gca()
# axes.set_ylim([0,0.1])
plt.legend()
# plt.show()
plt.savefig(f'./Graphs_modified/{gname}_Valloss_.png', dpi=600)
