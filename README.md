This CheckersAPI is hosted in Google Cloud and is what our checkers game found at 
https://checkersgame-a3b92.web.app/ calls when the AI makes its move. App.py is where the logic of the API is hosted
The MCTS AI was made using classes for the checkers game defining the game logic. The node class defines the 
attributes for the nodes used with the MCTS class to search possible moves to decide which one is best with the UCB 
forumla. The ResNet class is what is used to upload the model that we trained on over 20,000 games the AI played 
against itself and was trained on and is the architechure we used for our CNN. The A-B pruning alrogithm searches
all possible moves and uses a heuristic to decide which move is best and scores each possible state to optimize the
searching algorithm to only include good moves. The other files define what modules are needed to run this in the
API, PyTorch and Numpy, and how the repository should be uploaded into Google Cloud
