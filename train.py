from game import Game
from model import ModelDenseSigmoidMasked, ModelBase
from keras.models import Model
import logging
import numpy as np
import os
import json
import shutil

class Train:
    def __init__(self, modelO : ModelBase, modelX : ModelBase, batch=100):
        np.random.seed(0)
        self.modelO = modelO
        self.modelX = modelX
        self.batch = batch
        self.callbacks = []
        self.trainingInfo = {
            'trained_epochs': 0,
            'batch': batch,
            'modelO': { 'loss': 0, 'weights' : '' },
            'modelX': { 'loss': 0, 'weights' : '' },
            'totalLoss': 0
        }

    def UpdateTrainingInfo(self, e, lossO, lossX, OWeights, XWeights):
        self.trainingInfo['trained_epochs'] = e
        self.trainingInfo['modelO']['loss'] = lossO
        self.trainingInfo['modelO']['weights'] = OWeights
        self.trainingInfo['modelX']['loss'] = lossX
        self.trainingInfo['modelX']['weights'] = XWeights
        self.trainingInfo['totalLoss'] = lossO + lossX

    def Train(self, epochs, modelOWeights=None, modelXWeights=None, jsonFile=None):
        weightFiles = []
        if modelOWeights:
            weightFiles.append(modelOWeights)
            if os.path.exists(modelOWeights):
                print("Load weights from %s" % modelOWeights)
                self.modelO.LoadWeights(modelOWeights)
            self.callbacks.append(lambda e, lossO, lossX : self.modelO.SaveWeights(modelOWeights))
        if modelXWeights:
            weightFiles.append(modelXWeights)
            if os.path.exists(modelXWeights):
                print("Load weights from %s" % modelXWeights)
                self.modelX.LoadWeights(modelXWeights)
            self.callbacks.append(lambda e, lossO, lossX : self.modelX.SaveWeights(modelXWeights))
        if jsonFile:
            if os.path.exists(jsonFile):
                print("Load trainingInfo from: %s" % jsonFile)
                self.trainingInfo = json.loads(open(jsonFile, "r").read())
                print(self.trainingInfo)

            self.callbacks.append(lambda e, lossO, lossX: self.UpdateTrainingInfo(e, lossO, lossX, modelOWeights, modelXWeights))
            self.callbacks.append(lambda _, __, ___ : open(jsonFile, "w").write(json.dumps(self.trainingInfo)))


        starting_epochs = self.trainingInfo['trained_epochs'] + 1

        for e in range(starting_epochs, epochs + starting_epochs):
            print("Epoch %d: " % e)
            lo, lx = self.TrainEpoch(e)
            print("Callbacks...")
            if e > 1:
                for weights in weightFiles:
                    shutil.copy(weights, "backup_" + weights)
            for callback in self.callbacks:
                callback(e, lo, lx)

    def TrainEpoch(self, e):
        games = [Game.GetRandomGame(e) for i in range(self.batch)]
        OIndexes = [i for i in range(self.batch) if games[i].GetCurrentPlayer() == Game.O]
        XIndexes = [i for i in range(self.batch) if games[i].GetCurrentPlayer() == Game.X]

        xInput = np.stack([g.AsVector() for g in games], axis=0)
        yTrue = self.PlayGames(games)

        trainedModel : Model = self.modelO.GetModel()
        trainedModel.train_on_batch(x=xInput[OIndexes], y=yTrue[OIndexes])
        trainingLossO = trainedModel.evaluate(xInput[OIndexes], yTrue[OIndexes])
        print("modelO: training_loss: %f" % trainingLossO)

        trainedModel : Model = self.modelX.GetModel()
        trainedModel.train_on_batch(x=xInput[XIndexes], y=yTrue[XIndexes])
        trainingLossX = trainedModel.evaluate(xInput[XIndexes], yTrue[XIndexes])
        print("modelX: training_loss: %f" % trainingLossX)

        print("total loss: %f" % (trainingLossO + trainingLossX))

        if e % 3 == 0:
            print("Example of O: ")
            exampleGame = games[OIndexes[0]]
            print(exampleGame)
            print(np.resize(self.GetPrediction(self.modelO, exampleGame), (Game.ROW_COUNT, Game.COL_COUNT)))
            print(np.resize(yTrue[OIndexes[0]], (Game.ROW_COUNT, Game.COL_COUNT)))

            print("Example of X: ")
            exampleGame = games[XIndexes[0]]
            print(exampleGame)
            print(np.resize(self.GetPrediction(self.modelX, exampleGame), (Game.ROW_COUNT, Game.COL_COUNT)))
            print(np.resize(yTrue[XIndexes[0]], (Game.ROW_COUNT, Game.COL_COUNT)))
        if e % 20 == 0:
            print("Game play: ")
            def GameCallback(g):
                print("----------------")
                print(g)
                model = { Game.O : self.modelO, Game.X : self.modelX }[g.GetCurrentPlayer()]
                print(np.reshape(self.GetPrediction(model, g), (Game.ROW_COUNT, Game.COL_COUNT)))

            self.PlayGame(Game(), GameCallback)


        return trainingLossO, trainingLossX

    def PlayGames(self, games):
        y_true = np.zeros((self.batch, Game.BOARD_SIZE))
        for i, game in enumerate(games):
            # For each game, calculate y_true
            # which means we will try Game.BOARD_SIZE places and find out whether we will win.
            for pos in range(Game.BOARD_SIZE):
                x, y = Game.PosToXY(pos)
                if game.IsEmpty(x, y):
                    trainedPlayer = game.GetCurrentPlayer()
                    curGame = game.clone()
                    curGame.Play(trainedPlayer, x, y)
                    gameResult, steps = self.PlayGame(curGame)
                    if gameResult == trainedPlayer:
                        gameResult = (1 - 0.7) * ((0.5) ** (2 * (steps - 1) / (Game.BOARD_SIZE - game.steps))) + 0.7
                    else:
                        gameResult = ((steps - 1) / (Game.BOARD_SIZE - game.steps)) * 0.4
                    y_true[i, pos] = gameResult

        return y_true

    def PlayGame(self, game : Game, cb = None):
        winner = game.GetWinner()
        steps = 1
        while not game.IsFull():
            winner = game.GetWinner()
            if winner != Game.EMPTY:
                break
            if game.GetCurrentPlayer() == Game.O:
                x, y = self.GetNextBestMove(self.modelO, game)
            else:
                x, y = self.GetNextBestMove(self.modelX, game)
            game.Play(game.GetCurrentPlayer(), x, y)
            steps += 1
            if cb:
                cb(game.clone())

        return winner, steps


    @staticmethod
    def GetPrediction(model : ModelBase, game : Game):
        # print(np.reshape(model.Predict(np.expand_dims(game.AsVector(), 0))[0, :], (3,3)))
        return model.Predict(np.expand_dims(game.AsVector(), 0))[0, :]

    @staticmethod
    def GetNextMove(model : ModelBase, game : Game, selector):
        # Inputs:
        #   Current game status
        # Outputs:
        #   (row, col) predicted by the model
        prediction = Train.GetPrediction(model, game)
        maxPos = selector(prediction)
        return Game.PosToXY(maxPos)

    @staticmethod
    def GetNextBestMove(model : ModelBase, game : Game):
        return Train.GetNextMove(model, game, np.argmax)

    @staticmethod
    def GetNextWorstMove(model : ModelBase, game : Game):
        return Train.GetNextMove(model, game, np.argmin)






