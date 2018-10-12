from game import Game

import timeit

def TestGame():
    print(Game().GetBoard())


    assert(Game().GetWinner() == Game.EMPTY)

    game = Game()
    game.Play(Game.O, 0, 0)
    game.Play(Game.O, 0, 1)
    game.Play(Game.O, 0, 2)
    print(game)
    assert (game.GetWinner() == Game.O)

    game = Game()
    game.Play(Game.X, 1, 0)
    game.Play(Game.X, 0, 0)
    game.Play(Game.X, 2, 0)
    print(game)
    assert (game.GetWinner() == Game.X)

    game = Game()
    game.Play(Game.X, 0, 0)
    game.Play(Game.X, 1, 1)
    game.Play(Game.X, 2, 2)
    print(game)
    assert (game.GetWinner() == Game.X)

    game = Game()
    game.Play(Game.O, 0, 0)
    game.Play(Game.O, 1, 1)
    game.Play(Game.O, 2, 2)
    print(game)
    assert (game.GetWinner() == Game.O)

    game = Game()
    game.Play(Game.O, 2, 0)
    game.Play(Game.O, 1, 1)
    game.Play(Game.O, 0, 2)
    print(game)
    assert (game.GetWinner() == Game.O)

    game = Game()
    game.Play(Game.O, 1, 0)
    game.Play(Game.O, 1, 1)
    game.Play(Game.O, 1, 2)
    print(game)
    assert (game.GetWinner() == Game.O)

    game = Game()
    game.Play(Game.O, 0, 0)
    game.Play(Game.O, 0, 2)
    game.Play(Game.O, 1, 2)
    game.Play(Game.X, 1, 0)
    game.Play(Game.X, 1, 1)
    game.Play(Game.X, 2, 1)
    game.Play(Game.O, 2, 2)
    print(game)
    assert (game.GetWinner() == Game.O)


    def RandomGames():
        game = Game.GetRandomGame()
        assert (game.GetWinner() == Game.EMPTY)
        return game

    for i in range(1000):
        RandomGames()

    for i in range(4):
        print(RandomGames())

    print(Game().AsVector())

    game = Game()
    assert(not game.IsFull())
    for i in range(3):
        for j in range(3):
            game.Play(Game.O, i, j)
    assert(game.IsFull())

    game = Game()
    game.Play(Game.O, 0, 2)
    game.Play(Game.O, 1, 2)
    game.Play(Game.O, 2, 2)
    print(game)
    assert (game.GetWinner() == Game.O)

    game = Game()
    game.Play(Game.X, 0, 2)
    game.Play(Game.X, 1, 2)
    game.Play(Game.X, 2, 2)
    print(game)
    assert (game.GetWinner() == Game.X)


def TestModel():
    from model import ModelDenseSigmoidMasked
    ModelDenseSigmoidMasked()


TestGame()
# TestModel()