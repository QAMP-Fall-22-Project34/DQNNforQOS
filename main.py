'''
File: main.py
Project: DQNNforQOS
Created Date: Fri Oct 28 2022 07:06:04 AM +09:00
Author: Siheon Park (sihoney97@kaist.ac.kr)
-----
Last Modified: Fr Oct 28 2022 07:12:28 AM +09:00
Modified By: Siheon Park (sihoney97@kaist.ac.kr)
-----
Copyright (c) 2022 Siheon Park
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''
from dqnn.dataset import generate_random_training_data, generate_target_from_unitary
from dqnn.opflow import OpflowDQNN, random_unitary
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

if __name__ == "__main__":
    unitary = random_unitary(2**2)
    model = OpflowDQNN([2, 3, 2], epsilon=0.1)
    unitaries = model.makeRandomUnitaries()
    trainingData = generate_random_training_data(10, 2)
    targetStates = generate_target_from_unitary(unitary, trainingData)

    loss_list = []
    writer = SummaryWriter(log_dir='./runs/DQNN')
    for s in trange(100):
        loss = model.step(unitaries, trainingData, targetStates)
        writer.add_scalar('Loss', loss, s)
        loss_list.append(loss)
        if loss>0.99:
            break

    plt.plot(loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Cost/Fidelity')
    plt.show()