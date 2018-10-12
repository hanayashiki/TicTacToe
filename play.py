from game import Game
from train import Train


def DefaultInputter():
    keyToPos = {
        'Q': 0, 'W': 1, 'E': 2,
        'A': 3, 'S': 4, 'D': 5,
        'Z': 6, 'X': 7, 'C': 8
    }
    while True:
        key = input("key >>").upper()
        if not key in keyToPos:
            continue
        else:
            return keyToPos[key]

def DefaultDisplayer(game : Game):
    print()
    print(game)
    print()

class Play:

    def __init__(self, AIModel, playerRole, inputter=DefaultInputter, displayer=DefaultDisplayer):
        self.aiModel = AIModel
        self.playerRole = playerRole
        self.inputter = inputter
        self.displayer = displayer

    def Start(self):
        self.game = Game()
        while not self.game.IsFull():
            self.displayer(self.game)
            if self.playerRole == self.game.GetCurrentPlayer():
                while True:
                    inputPos = self.inputter()
                    x, y = self.game.PosToXY(inputPos)
                    if self.game.IsEmpty(x, y):
                        break
                    else:
                        print("Place already taken up")
                        continue
            else:
                x, y = Train.GetNextBestMove(self.aiModel, self.game)

            self.game.Play(self.game.GetCurrentPlayer(), x, y)

            if self.game.GetWinner() != Game.EMPTY:
                self.displayer(self.game)
                print("Player %s wins ! " % Game.PLAYER_ATTR[self.game.GetWinner()]['name'])
                break
            elif self.game.IsFull():
                self.displayer(self.game)
                print("It's a tie !")
                break


