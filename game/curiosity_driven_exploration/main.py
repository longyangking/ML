import numpy as np 
import argparse

__channel_size__ = 4
__gym_game_names__ = {
    "Acrobot":"Acrobot-v1",
    "CartPole":"CartPole-v1",
    "MountainCar":"MountainCar-v0",
    "Pendulum":"Pendulum-v0",
    "MountainCarContinuous":"MountainCarContinuous-v0"
}

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__info__)
    parser.add_argument("--retrain", action='store_true', default=False, help="Re-Train AI")
    parser.add_argument("--train",  action='store_true', default=False, help="Train AI")
    parser.add_argument("--verbose", action='store_true', default=False, help="Verbose")
    parser.add_argument("--play", action='store_true', default=False, help="Play by AI")
    parser.add_argument("--name", default="Acrobot", help='Game name')

    args = parser.parse_args()
    verbose = args.verbose
    game_name = args.name
    __filename__ = "model_{0}.h5".format(game_name)

    if game_name not in __gym_game_names__.keys():
        print("Error in game name: Not supported game yet. [{0}]".format(game_name))
        exit()

    gym_game_name = __gym_game_names__[game_name]

    if args.train:
        if verbose:
            print("Continue to train AI model for game: [{0}].".format(game_name))

        from ai import AI
        from train import TrainAI

        ai = AI(game_name=gym_game_name, channel=__channel_size__, verbose=verbose)
        if verbose:
            print("loading latest model: [{0}] ...".format(__filename__),end="")
        ai.load_nnet(__filename__)
        if verbose:
            print("load OK!")

        trainai = TrainAI(
            game_name=gym_game_name,
            channel=__channel_size__,
            ai=ai,
            verbose=verbose
        )
        trainai.start(filename=__filename__)

        if verbose:
            print("The latest AI model is saved as [{0}]".format(__filename__))

    if args.retrain:
        if verbose:
            print("Start to re-train AI model for game: [{0}].".format(game_name))

        from train import TrainAI

        trainai = TrainAI(game_name=gym_game_name, channel=__channel_size__, verbose=verbose)
        trainai.start(__filename__)

        if verbose:
            print("The latest AI model is saved as [{0}]".format(__filename__))

    if args.play:
        if verbose:
            print("Start to play the game by the AI model, which will be rendered in the screen.")

        from ai import AI
        from game import GameEngine

        ai = AI(game_name=gym_game_name, verbose=verbose)
        if verbose:
            print("loading latest model: [{0}] ...".format(__filename__),end="")
        ai.load_nnet(__filename__)
        if verbose:
            print("load OK!")

        print("Please close game in terminal after closing window (i.e, Press Ctrl+C).")
        engine = GameEngine(game_name=gym_game_name, ai=ai, verbose=verbose)
        engine.start()