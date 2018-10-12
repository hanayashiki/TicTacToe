from train import Train
from model import ModelDenseSigmoidMasked, ModelTwoDensesSigmoidMasked, \
    ModelTwoDensesReluSigmoidMasked, ModelThreeDensesReluSigmoidMasked, ModelThreeDensesReluReluAdamMasked
import pickle

if __name__ == '__main__':
    def Train1():
        modelO = ModelDenseSigmoidMasked()
        modelX = ModelDenseSigmoidMasked()
        trainer = Train(modelO, modelX)
        trainer.Train(1000, "O1.h5", "X1.h5", "info1.json")

    def Train2():
        modelO = ModelTwoDensesSigmoidMasked()
        modelX = ModelTwoDensesSigmoidMasked()
        trainer = Train(modelO, modelX)
        trainer.Train(5000, "O2.h5", "X2.h5", "info2.json")

    def Train3():
        modelO = ModelTwoDensesReluSigmoidMasked()
        modelX = ModelTwoDensesReluSigmoidMasked()
        trainer = Train(modelO, modelX)
        trainer.Train(1000, "O3.h5", "X3.h5", "info3.json")

    def Train4():
        modelO = ModelTwoDensesReluSigmoidMasked()
        modelX = ModelTwoDensesReluSigmoidMasked()
        trainer = Train(modelO, modelX)
        trainer.Train(1000, "O4.h5", "X4.h5", "info4.json")

    def Train5():
        modelO = ModelTwoDensesReluSigmoidMasked(midDim=27)
        modelX = ModelTwoDensesReluSigmoidMasked(midDim=27)
        trainer = Train(modelO, modelX)
        trainer.Train(1000, "O5.h5", "X5.h5", "info5-midDim=27.json")

    def Train6():
        modelO = ModelTwoDensesReluSigmoidMasked(midDim=128)
        modelX = ModelTwoDensesReluSigmoidMasked(midDim=128)
        trainer = Train(modelO, modelX, batch=64)
        trainer.Train(1000, "O6.h5", "X6.h5", "info6-midDim=128.json")

    def Train7():
        modelO = ModelThreeDensesReluSigmoidMasked(midDim1=27, midDim2=27)
        modelX = ModelThreeDensesReluSigmoidMasked(midDim1=27, midDim2=27)
        trainer = Train(modelO, modelX, batch=64)
        trainer.Train(1000, "O7.h5", "X7.h5", "info7-midDim1=27-midDim2=27.json")

    def Train8():
        modelO = ModelThreeDensesReluSigmoidMasked(midDim1=256, midDim2=256, loss='mean_squared_error')
        modelX = ModelThreeDensesReluSigmoidMasked(midDim1=256, midDim2=256, loss='mean_squared_error')
        trainer = Train(modelO, modelX, batch=32)
        trainer.Train(1000, "O8.h5", "X8.h5", "info8-midDim1=256-midDim2=256.json")

    def Train9():
        modelO = ModelThreeDensesReluReluAdamMasked(midDim1=256, midDim2=256, loss='mean_squared_error')
        modelX = ModelThreeDensesReluReluAdamMasked(midDim1=256, midDim2=256, loss='mean_squared_error')
        trainer = Train(modelO, modelX, batch=32)
        trainer.Train(2000, "O9.h5", "X9.h5", "info9-midDim1=256-midDim2=256-relu-adam.json")

    Train9()