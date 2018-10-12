from train import Train
from model import ModelTwoDensesSigmoidMasked, ModelDenseSigmoidMasked
from game import Game

if __name__ == '__main__':
    def TestGame1():
        modelO = ModelTwoDensesSigmoidMasked()
        modelX = ModelTwoDensesSigmoidMasked()
        trainer = Train(modelO, modelX)
        modelO.LoadWeights('O2.h5')
        modelX.LoadWeights('X2.h5')
        trainer.PlayGame(Game(), lambda g: print(g))

    def TestGame2():
        modelO = ModelDenseSigmoidMasked()
        modelX = ModelDenseSigmoidMasked()
        trainer = Train(modelO, modelX)
        modelO.LoadWeights('O1.h5')
        modelX.LoadWeights('X1.h5')
        trainer.PlayGame(Game(), lambda g: print(g))

    def TestGame3():
        modelO = ModelDenseSigmoidMasked()
        modelX = ModelTwoDensesSigmoidMasked()
        trainer = Train(modelO, modelX)
        modelO.LoadWeights('O1.h5')
        modelX.LoadWeights('X2.h5')
        trainer.PlayGame(Game(), lambda g: print(g))

    TestGame3()