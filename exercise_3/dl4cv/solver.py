from random import shuffle
import numpy as np
from torch import nn
import torch.nn.functional as F

import torch
from torch.autograd import Variable

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)

class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        #self.loss_func = loss_func
        self.loss_func = CrossEntropyLoss2d(size_average=False, ignore_index=-1)

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        if torch.cuda.is_available():
            print('cuda available')
            model.cuda()

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            last_train_acc = 0
            
            correct_train = 0
            total_train = 0
            for i, data in enumerate(train_loader, 0):
                # get the inputs
                inputs, labels = data
                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(labels)
                #print("hurray")
                #print("inputs1.data = ", np.unique(inputs.data[0]))
                #print("labels1.data = ", np.unique(labels.data[0]))

                # zero the parameter gradients
                optim.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                #print("VKP size", outputs.size)
                #print("output ", outputs)
                

                loss = self.loss_func(outputs, labels)
                loss.backward()
                optim.step()

                # print statistics
                #running_loss += loss.data[0]
                #self.train_loss_history.append(loss.data[0])
                #_, predicted = torch.max(outputs.data, 1)
                #total_train += labels.size(0)
                #print("predicted")
                #print(predicted)
                #print("labels")
                #print(labels)
                #correct_train += (predicted == labels.data).sum()
                #if i % log_nth == 0:
                print('[epoch %d Iteration %d/%d] TRAIN loss:  %.3f'%(epoch, i, iter_per_epoch, loss.data[0]))
                #train_accuracy = correct_train/total_train
                #last_train_acc = train_accuracy
            """      
            self.train_acc_history.append(last_train_acc)
            print('[epoch %d/%d] TRAIN acc/loss:  %.3f/%.3f'%(epoch, num_epochs, last_train_acc, self.train_loss_history[-1]))
            
            ############Validation Accuracy at the end of each epoch#############
            correct_val = 0
            total_val = 0
            for data in val_loader:
                images_val, labels_val = data
                images_val, labels_val = Variable(images_val), Variable(labels_val)
                outputs_val = model(images_val)
                loss_val = self.loss_func(outputs_val, labels_val)
                _, predicted = torch.max(outputs_val.data, 1)
                total_val += labels_val.size(0)
                correct_val += (predicted == labels_val.data).sum()
            accuracy_val = correct_val/total_val
            print('[epoch %d/%d] VAL acc/loss:  %.3f/%.3f'%(epoch, num_epochs, accuracy_val, loss_val.data[0]))
            self.val_acc_history.append(accuracy_val)
            """


        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')