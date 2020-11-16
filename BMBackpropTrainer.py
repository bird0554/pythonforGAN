# coding=utf-8
from math import fabs

from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.utilities import fListToString


class BMBackpropTrainer(BackpropTrainer):
    def __init__(self, module, dataset=None, learningrate=0.01, lrdecay=1.0,
                 momentum=0., verbose=False, batchlearning=False,
                 weightdecay=0.):
        BackpropTrainer.__init__(self, module, dataset, learningrate, lrdecay, momentum, verbose, batchlearning)

    def bmtrain(self, dataset=None, maxEpochs=None, verbose=None, continueEpochs=1000, totalError=0.00001):
        epochs = 0
        if dataset == None:
            dataset = self.ds
        if verbose == None:
            verbose = self.verbose
        # Split the dataset randomly: validationProportion of the samples for
        # validation.
        bestweights = self.module.params.copy()
        bestverr = self.testOnData(self.ds)
        trainingErrors = []
        # validationErrors = [bestverr]
        while True:
            trainingErrors.append(self.train())
            # validationErrors.append(self.testOnData(validationData))
            if epochs == 0 or trainingErrors[-1] < bestverr:
                # one update is always done
                bestverr = trainingErrors[-1]
                bestweights = self.module.params.copy()

            if maxEpochs != None and epochs >= maxEpochs:
                self.module.params[:] = bestweights
                break
            epochs += 1

            if len(trainingErrors) >= continueEpochs * 2:
                # have the validation errors started going up again?
                # compare the average of the last few to the previous few
                old = trainingErrors[-continueEpochs * 2:-continueEpochs]
                new = trainingErrors[-continueEpochs:]
                if (fabs(min(new) - max(old)) < totalError):
                    self.module.params[:] = bestweights
                    break
        # trainingErrors.append(self.testOnData(trainingData))
        self.ds = dataset
        if verbose:
            print('train-errors:', fListToString(trainingErrors, 6))
            # print 'valid-errors:', fListToString(validationErrors, 6)
        return trainingErrors

