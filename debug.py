from train import Train
from model import ModelThreeDensesReluSigmoidMasked, ModelDenseSigmoidMasked
from game import Game

if __name__ == '__main__':

    modelO = ModelThreeDensesReluSigmoidMasked(midDim1=27, midDim2=27)
    modelX = ModelThreeDensesReluSigmoidMasked(midDim1=27, midDim2=27)
    trainer = Train(modelO, modelX)
    modelO.LoadWeights('O7.h5')
    modelX.LoadWeights('X7.h5')

    game = Game()
    game.Play(Game.O, 0, 0)
    game.Play(Game.O, 1, 1)
    game.Play(Game.X, 0, 1)
    game.Play(Game.X, 1, 0)
    print(game)

    result, steps = trainer.PlayGame(game, lambda g: print(g))
    print("result: %f, steps: %d" % (result, steps))

    print("----")

    game = Game()
    game.Play(Game.O, 0, 1)
    game.Play(Game.O, 1, 1)
    game.Play(Game.O, 1, 2)
    game.Play(Game.X, 0, 0)
    game.Play(Game.X, 1, 0)
    game.Play(Game.X, 2, 1)
    game.Play(Game.X, 2, 2)

    print(game)

    result, steps = trainer.PlayGame(game, lambda g: print(g))
    print("result: %f, steps: %d" % (result, steps))

    game = Game()
    game.Play(Game.O, 0, 1)
    game.Play(Game.O, 0, 2)
    game.Play(Game.O, 1, 2)
    game.Play(Game.X, 0, 0)
    game.Play(Game.X, 1, 1)

    print(game)

    result, steps = trainer.PlayGame(game, lambda g: print(g))
    print("result: %f, steps: %d" % (result, steps))
