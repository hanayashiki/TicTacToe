from play import Play
from model import ModelTwoDensesSigmoidMasked, ModelDenseSigmoidMasked, ModelThreeDensesReluSigmoidMasked, \
    ModelThreeDensesReluReluAdamMasked
from game import Game


if __name__ == '__main__':
    modelO = ModelThreeDensesReluReluAdamMasked(midDim1=256, midDim2=256, loss='mean_squared_error')
    modelX = ModelThreeDensesReluReluAdamMasked(midDim1=256, midDim2=256, loss='mean_squared_error')
    modelO.LoadWeights('O9.h5')
    modelX.LoadWeights('X9.h5')

    while True:
        while True:
            x = input(">> Select your role: 'X' or 'O'. ").upper()
            if x == 'X':
                aiModel = modelO
                break
            elif x == 'O':
                aiModel = modelX
                break
            else:
                continue

        playerRole = Game.NAME_TO_PLAYER[x]

        play = Play(aiModel, playerRole)
        play.Start()

        print("\nNew game!")



