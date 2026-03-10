from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
#  GAME + MODEL (copied from training code)
# ─────────────────────────────────────────────
class Checkers:
    def __init__(self):
        self.rowCount = 8
        self.colCount = 8
        self.actionSize = 4096
        self.kingDirections = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
        self.goodDirections = [[-1, 1], [-1, -1]]
        self.evilDirections = [[1, 1], [1, -1]]

    def getLegalMoves(self, state, r, c, player, isChainJump=False, visited=None):
        if visited is None:
            visited = []
        piece = state[r, c]
        isKing = abs(piece) == 2
        if isKing:
            directions = self.kingDirections
        elif player == 1:
            directions = self.goodDirections
        else:
            directions = self.evilDirections
        jumps = []
        slides = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            jr, jc = r + dr * 2, c + dc * 2
            if 0 <= jr < 8 and 0 <= jc < 8:
                victim = state[nr, nc]
                land = state[jr, jc]
                if victim != 0 and np.sign(victim) != player and land == 0:
                    if (nr, nc) not in visited:
                        nextVisited = visited + [(nr, nc)]
                        willBeKing = isKing or (player == 1 and jr == 0) or (player == -1 and jr == 7)
                        tempState = np.copy(state)
                        tempState[jr, jc] = player * (2 if willBeKing else 1)
                        tempState[r, c] = 0
                        tempState[nr, nc] = 0
                        extraJumps = self.getLegalMoves(tempState, jr, jc, player, True, nextVisited)
                        if extraJumps['type'] == 'jump':
                            for ej in extraJumps['possible']:
                                jumps.append((r, c, ej[2], ej[3]))
                        else:
                            jumps.append((r, c, jr, jc))
            if not isChainJump and 0 <= nr < 8 and 0 <= nc < 8:
                if state[nr, nc] == 0:
                    slides.append((r, c, nr, nc))
        if jumps:
            return {'possible': list(set(jumps)), 'type': 'jump'}
        return {'possible': slides, 'type': 'slide'}

    def getValidMoves(self, state, player):
        validMoves = np.zeros(self.actionSize)
        allJumps = []
        allSlides = []
        for r in range(8):
            for c in range(8):
                if np.sign(state[r, c]) == player:
                    moves = self.getLegalMoves(state, r, c, player)
                    if moves['type'] == 'jump':
                        allJumps.extend(moves['possible'])
                    else:
                        allSlides.extend(moves['possible'])
        actualMoves = allJumps if allJumps else allSlides
        for r1, c1, r2, c2 in actualMoves:
            actionId = (r1 * 8 + c1) * 64 + (r2 * 8 + c2)
            validMoves[actionId] = 1
        return validMoves

    def getNextState(self, state, action, player):
        start_sq, end_sq = divmod(action, 64)
        r1, c1 = divmod(start_sq, 8)
        r2, c2 = divmod(end_sq, 8)
        next_state = np.copy(state)
        piece = next_state[r1, c1]
        dr = int(np.sign(r2 - r1))
        dc = int(np.sign(c2 - c1))
        curr_r, curr_c = r1 + dr, c1 + dc
        while (curr_r, curr_c) != (r2, c2):
            if np.sign(next_state[curr_r, curr_c]) == -player:
                next_state[curr_r, curr_c] = 0
            curr_r += dr
            curr_c += dc
        next_state[r2, c2] = piece
        next_state[r1, c1] = 0
        if (player == 1 and r2 == 0) or (player == -1 and r2 == 7):
            next_state[r2, c2] = player * 2
        return next_state
        
    def getValueAndTerminated(self, state, player, spg=None):
        if not np.any(np.sign(state) == -player):
            return 1, True
        if np.sum(self.getValidMoves(state, -player)) == 0:
            return 1, True

        if np.sum(self.getValidMoves(state, player)) == 0:
            return -1, True
        if spg is not None:
            # 1. Insufficient material: 1 king vs 1 king
            pieces = state.flatten()
            playerPieces   = pieces[pieces * player > 0]
            opponentPieces = pieces[pieces * player < 0]
            if (len(playerPieces) == 1 and playerPieces[0] == player * 2 and
                len(opponentPieces) == 1 and opponentPieces[0] == -player * 2):
                return 0, True

            # 2. 80-move rule (40 per player = 80 half-moves)
            if spg.movesSinceAdvancement >= 80:
                return 0, True

            # 3. Three-fold repetition
            boardHash = state.tobytes()
            spg.positionHistory[boardHash] = spg.positionHistory.get(boardHash, 0) + 1
            if spg.positionHistory[boardHash] >= 3:
                return 0, True

        return 0, False

    def getOpponentValue(self, value):
        return -value

    def changePerspective(self, state, player):
        if player == -1:
            return np.flip(state, axis=0) * -1
        return state

    def getEncodedState(self, state):
        state = np.array(state)
        if state.ndim == 2:
            state = state[np.newaxis]
        encoded = np.stack([
            state == -2,
            state == -1,
            state == 0,
            state == 1,
            state == 2,
        ], axis=1).astype(np.float32)
        return encoded


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__()
        self.device = device
        self.startBlock = nn.Sequential(
            nn.Conv2d(5, num_hidden, 3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        self.backBone = nn.ModuleList([ResBlock(num_hidden) for _ in range(num_resBlocks)])
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.rowCount * game.colCount, game.actionSize)
        )
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, 3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.rowCount * game.colCount, 1),
            nn.Tanh()
        )
        self.to(device)

    def forward(self, x):
        x = self.startBlock(x)
        for block in self.backBone:
            x = block(x)
        return self.policyHead(x), self.valueHead(x)


class Node:
    def __init__(self, game, args, state, parent=None, actionTaken=None, prior=0, visitCount=0):
        self.game        = game
        self.args        = args
        self.state       = state
        self.parent      = parent
        self.actionTaken = actionTaken
        self.prior       = prior
        self.children    = []
        self.visitCount  = visitCount
        self.valueSum    = 0

    def isFullyExpanded(self):
        return len(self.children) > 0

    def select(self):
        bestChild, bestUCB = None, -np.inf
        for child in self.children:
            ucb = self.getUCB(child)
            if ucb > bestUCB:
                bestChild, bestUCB = child, ucb
        return bestChild

    def getUCB(self, child):
        qValue = 0 if child.visitCount == 0 else \
                 1 - ((child.valueSum / child.visitCount) + 1) / 2
        return qValue + self.args['C'] * (math.sqrt(self.visitCount) / (child.visitCount + 1)) * child.prior

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                childState = self.state.copy()
                childState = checkers.getNextState(childState, action, 1)
                childState = checkers.changePerspective(childState, player=-1)
                self.children.append(Node(checkers, self.args, childState, self, action, prob))

    def backpropogate(self, value):
        self.valueSum  += value
        self.visitCount += 1
        if self.parent is not None:
            self.parent.backpropogate(-value)


# ─────────────────────────────────────────────
#  LOAD MODEL ON STARTUP
# ─────────────────────────────────────────────
checkers = Checkers()
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(checkers, 4, 64, device)

MODEL_PATH = os.environ.get("MODEL_PATH", "model.pt")
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Loaded model from {MODEL_PATH}")
else:
    print("WARNING: No model file found, using untrained model")

model.eval()

MCTS_ARGS = {
    'C':                 2,
    'numSearches':       200,   # lower = faster response, higher = stronger play
    'dirichlet_epsilon': 0.0,   # no noise during inference
    'dirichlet_alpha':   0.3,
}


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def pieces_to_board(pieces):
    """Convert React piece list to numpy board.
    pieces = [{"r": 0, "c": 1, "isKing": false, "team": "evil"}, ...]
    team "good" = player 1, team "evil" = player -1
    """
    board = np.zeros((8, 8), dtype=int)
    for p in pieces:
        r, c   = p['r'], p['c']
        player = 1 if p['team'] == 'good' else -1
        value  = 2 if p['isKing'] else 1
        board[r, c] = player * value
    return board


def action_to_move(action):
    """Convert action int to {r1, c1, r2, c2} dict for React."""
    start_sq, end_sq = divmod(action, 64)
    r1, c1 = divmod(start_sq, 8)
    r2, c2 = divmod(end_sq, 8)
    return {'r1': int(r1), 'c1': int(c1), 'r2': int(r2), 'c2': int(c2)}


def get_best_move(board, player=-1):
    """Run MCTS and return the best action."""
    # Always think from current player's perspective
    neutralState = checkers.changePerspective(board, player)

    encoded = checkers.getEncodedState(neutralState)
    with torch.no_grad():
        policy, _ = model(
            torch.tensor(encoded, dtype=torch.float32, device=device)
        )
        policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()

    validMoves = checkers.getValidMoves(neutralState, 1)  # always 1 after perspective flip
    policy *= validMoves
    psum = np.sum(policy)
    if psum > 0:
        policy /= psum

    # Run MCTS from root
    root = Node(checkers, MCTS_ARGS, neutralState, visitCount=1)
    root.expand(policy)

    for _ in range(MCTS_ARGS['numSearches']):
        node = root
        while node.isFullyExpanded():
            node = node.select()

        value, isTerminate = checkers.getValueAndTerminated(node.state, 1)

        if not isTerminate:
            encoded = checkers.getEncodedState(node.state)
            with torch.no_grad():
                nodePol, nodeVal = model(
                    torch.tensor(encoded, dtype=torch.float32, device=device)
                )
            nodePol = torch.softmax(nodePol, dim=1).squeeze(0).cpu().numpy()
            nodeVal = nodeVal.item()

            valid = checkers.getValidMoves(node.state, 1)
            nodePol *= valid
            psum = np.sum(nodePol)
            if psum > 0:
                nodePol /= psum

            node.expand(nodePol)
            value = nodeVal

        node.backpropogate(value)

    # Pick action with highest visit count
    visitCounts = np.array([child.visitCount for child in root.children])
    bestChild   = root.children[np.argmax(visitCounts)]
    return bestChild.actionTaken


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────
@app.route('/move', methods=['POST'])
def get_move():
    """
    Expects JSON body:
    {
        "pieces": [
            {"r": 0, "c": 1, "isKing": false, "team": "evil"},
            ...
        ]
    }

    Returns:
    {
        "move": {"r1": 5, "c1": 2, "r2": 4, "c2": 3},
        "action": 2738
    }
    """
    try:
        data   = request.get_json()
        pieces = data.get('pieces', [])

        if not pieces:
            return jsonify({'error': 'No pieces provided'}), 400

        board  = pieces_to_board(pieces)
        action = get_best_move(board, player=-1)  # AI is always -1
        move   = action_to_move(action)

        return jsonify({'move': move, 'action': int(action)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'device': str(device)})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
