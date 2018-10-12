import numpy as np
import random

class Game():
    ROW_COUNT = 3
    COL_COUNT = 3
    BOARD_SIZE = 9
    DIM_PER_BOX = 3
    # 9 * 3 matrix, where 9 is the count of boxes
    # In each box, [1, 0, 0] means O, [0, 1, 0] means empty, [0, 0, 1] means X
    MARK_O = np.array([1, 0, 0])
    MARK_EMPTY = np.array([0, 1, 0])
    MARK_X = np.array([0, 0, 1])

    O = 1
    EMPTY = 0.5
    X = 0

    NAME_TO_PLAYER = {
        'O': O,
        'X': X,
        '_': EMPTY
    }

    PLAYER_ATTR = {
        O: {
            'name': 'O',
            'mark': MARK_O
        },
        X: {
            'name': 'X',
            'mark': MARK_X
        },
        EMPTY: {
            'name': '_',
            'mark': MARK_EMPTY
        }
    }

    def __init__(self):
        self.board = np.expand_dims(self.MARK_EMPTY, 0)
        self.board = np.expand_dims(self.board, 0)
        self.board = np.repeat(self.board, self.COL_COUNT, axis=1)
        self.board = np.repeat(self.board, self.ROW_COUNT, axis=0)
        self.steps = 0

    def GetBoard(self):
        return self.board

    def GetStep(self):
        return self.steps

    def GetCurrentPlayer(self):
        if self.steps % 2 == 0:
            return self.O
        else:
            return self.X

    def AsVector(self):
        return np.reshape(self.board, (-1))

    def GetWinner(self):
        for row in range(0, self.ROW_COUNT):
            sumRow = np.sum(self.board[row, :, :], axis=0)
            if (sumRow == self.ROW_COUNT * self.MARK_O).all():
                return self.O
            elif (sumRow == self.ROW_COUNT * self.MARK_X).all():
                return self.X

        for col in range(0, self.COL_COUNT):
            sumColumn = np.sum(self.board[:, col, :], axis=0)
            if (sumColumn == self.COL_COUNT * self.MARK_O).all():
                return self.O
            elif (sumColumn == self.COL_COUNT * self.MARK_X).all():
                return self.X

        min_count = min(self.ROW_COUNT, self.COL_COUNT)
        diag1 = np.sum([ self.board[i, i, :] for i in range(min_count) ], axis=0)
        diag2 = np.sum([ self.board[i, min_count - i - 1, :] for i in range(min_count) ], axis=0)

        for diag in [diag1, diag2]:
            if (diag == min_count * self.MARK_O).all():
                return self.O
            if (diag == min_count * self.MARK_X).all():
                return self.X

        return self.EMPTY

    def IsFull(self):
        return np.sum(self.board, axis=(0, 1))[1] == 0

    def IsEmpty(self, row, col):
        return (self.board[row, col, :] == self.MARK_EMPTY).all()

    def Play(self, player, row, col):
        assert self.IsEmpty(row, col)
        self.steps += 1
        self.board[row, col, :] = self.PLAYER_ATTR[player]['mark']

    def Empty(self, row, col):
        assert not self.IsEmpty(row, col)
        self.steps -= 1
        self.board[row, col, :] = self.MARK_EMPTY

    @staticmethod
    def GetRandomGame(epoch):
        game = Game()
        # if epoch < 100:
        #     if random.randint(0, 9) >= 3:
        #         stepCount = random.randint(7, Game.BOARD_SIZE - 1) # note that random.randint is [a, b]
        #     else:
        #         stepCount = random.randint(0, 6)  # note that random.randint is [a, b]
        # else:
        stepCount = random.randint(0, Game.BOARD_SIZE - 1)
        places = np.random.permutation(Game.BOARD_SIZE)
        for i in range(stepCount):
            if i % 2 == 0:
                game.Play(Game.O, *Game.PosToXY(places[i]))
            else:
                game.Play(Game.X, *Game.PosToXY(places[i]))
            if game.GetWinner() != Game.EMPTY:
                game.Empty(*Game.PosToXY(places[i]))
                return game
        return game

    @staticmethod
    def PosToXY(pos):
        return pos // Game.COL_COUNT, pos % Game.COL_COUNT

    def MarkToPlayer(self, mark : np.array):
        if (mark == self.MARK_O).all():
            return self.O
        elif (mark == self.MARK_EMPTY).all():
            return self.EMPTY
        else:
            return self.X

    def __str__(self):
        res = ""
        for i in range(self.ROW_COUNT):
            for j in range(self.COL_COUNT):
                player = self.MarkToPlayer(self.GetBoard()[i, j, :])
                res += self.PLAYER_ATTR[player]['name'] + " "
            res += "\n"

        return res

    def clone(self) :
        new = Game()
        new.steps = self.steps
        new.board = self.board.copy()
        return new




