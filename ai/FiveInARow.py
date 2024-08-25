from __future__ import annotations
from itertools import batched
import numpy as np
from collections import deque
import torch
import torch.nn.functional as F
from torch import optim
from torch import tensor, nn, Tensor
import gymnasium as gym
import os

WHITE = 1
BLACK = -1

class FiveInARow(gym.Env):
    def __init__(self, rows=9, cols=9, player = WHITE):
        self.rows = rows
        self.cols = cols
        self.board = torch.zeros((rows,cols))
        self.player = player
        self.turn = 0
        self.winner = 0

        self.observation_space = gym.spaces.Box(-1,1,(rows,cols),np.int8)
        self.action_space = gym.spaces.Box(0,rows*cols - 1,dtype=np.int8)

    def pos_to_index(self, row, col):
        return row * self.cols + col

    def index_to_pos(self, index):
        return divmod(index, self.cols)

    def can_place(self, board, row, col):
        rows, cols = board.shape
        return row in range(rows) and col in range(cols) and board[row][col] == 0

    def place(self, board, row, col, player):
        new_board = board.detach().clone()
        new_board[row][col] = player
        return new_board

    def next_states(self, board, to_play):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.can_place(board, r, c):
                    yield self.pos_to_index(r, c), self.place(board, r, c, to_play)

    def count_pieces(self, board, row, col):
        vert = [(-1,0),(1,0)]
        horz = [(0,-1),(0,1)]
        posd = [(-1,1),(1,-1)]
        negd = [(1,1),(-1,-1)]

        piece = board[row][col]
        counts = []
        for dir in [vert,horz,posd,negd]:
            count = 1
            for dr,dc in dir:
                i = 1
                while (row + i*dr in range(self.rows)) and (col + i*dc in range(self.cols)) and (board[row + i*dr][col + i*dc] == piece):
                    count += 1
                    i += 1
            counts.append(count)
        return max(counts)

    def check_win(self, board, row, col):
        max_in_a_row = self.count_pieces(board, row, col)
        return max_in_a_row >= 5

    def get_winner(self, board):
        for r in range(self.rows):
            for c in range(self.cols):
                if board[r][c] != 0 and self.check_win(board, r, c):
                    return int(board[r][c])
        return 0

    def is_game_over(self, board):
        return (self.get_winner(board) != 0) or bool(torch.all(board != 0))

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.board = torch.zeros((self.rows,self.cols))
        self.player = WHITE
        return self.board, {}

    def step(self, action):
        row,col = self.index_to_pos(action)
        self.board = self.place(self.board,row,col,self.player)
        term = self.is_game_over(self.board)
        self.winner = self.get_winner(self.board)
        obs = self.board
        reward = self.winner
        self.player = -self.player
        return obs, reward, term, False, {}

class Model(nn.Module):
    def __init__(self, game_dim):
        super(Model, self).__init__()
        rows, cols = game_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=(3,3), padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3,3), padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3,3)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.3),
        )
        self.value_head = nn.Sequential(
            nn.Linear(128 * (rows-2) * (cols-2), 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(128 * (rows-2) * (cols-2), 256),
            nn.ReLU(),
            nn.Linear(256, rows * cols),
        )

    def forward(self, data: Tensor):
        x = self.cnn(data)
        value = self.value_head(x)
        logits = self.policy_head(x)
        return value, logits

class MCTS_Node:
    def __init__(self, prior, board: Tensor, to_play, parent = None, action = None):
        self.prior = prior
        self.board = board
        self.to_play = to_play
        self.parent = parent
        self.action = action
        self.predicted_value = None
        self.predicted_priors = None
        self.children: dict[int, MCTS_Node] = {}
        self.visits = 0
        self.value_sum = 0

    def value(self):
        return self.visits and self.value_sum / self.visits

    def uct_score(self, c_puct):
        parent = self.parent
        assert parent is not None
        Q_score = self.value()
        U_score = self.prior * c_puct * np.sqrt(parent.visits) / (1 + self.visits)
        return Q_score + U_score

    def get_data(self):
        data = torch.zeros((1,4,*self.board.shape))
        data[0, 0] = (self.board == self.to_play)
        data[0, 1] = (self.board == -self.to_play)
        _, cols = tuple(self.board.shape)
        if self.action is not None:
            data[0, 2, self.action // cols, self.action % cols] = 1
        data[0, 3, :, :] = max(self.to_play, 0)
        return data

    @property
    def expanded(self):
        return len(self.children) > 0

    def derived_policy(self, temperature = 1.0):
        assert self.predicted_priors is not None
        probs = torch.zeros_like(self.predicted_priors)
        for action,child in self.children.items():
            probs[action] = child.visits / self.visits
        logits = torch.log(probs)
        return F.softmax(logits / temperature, dim=0)

    def predict(self, model):
        if self.predicted_value is not None and self.predicted_priors is not None:
            return
        with torch.no_grad():
            pred_value, pred_logits = model(self.get_data())
            self.predicted_value = pred_value.flatten()
            masked_logits = pred_logits.flatten().where(self.board.flatten() == 0, float("-inf"))
            self.predicted_priors = F.softmax(masked_logits,dim=0)

    def apply_dirichlet_noise(self, d_alpha, d_epsilon):
        assert self.expanded
        legal_actions = [action for action,_ in self.children.items()]
        noise = np.random.dirichlet([d_alpha] * len(legal_actions))
        for action, n in zip(legal_actions, noise):
            self.children[action].prior = (1 - d_epsilon) * self.children[action].prior + d_epsilon * n

    def select(self, c_puct):
        node = self
        while node.expanded:
            node = max(node.children.values(), key=lambda n: n.uct_score(c_puct))
        return node

    def expand(self, game: FiveInARow, ignore_priors = False):
        assert ignore_priors or self.predicted_priors is not None
        if not self.expanded:
            for action, board in game.next_states(self.board, self.to_play):
                prior = self.predicted_priors[action] if not ignore_priors else 0
                self.children[action] = MCTS_Node(prior,board,-self.to_play,parent=self,action=action)

    def eval_and_backprop(self):
        assert self.predicted_value is not None
        value = self.predicted_value
        node = self
        while node is not None:
            node.value_sum += value
            node.visits += 1
            node = node.parent
            value = -value

    def __str__(self):
        children = sorted(list(self.children.values()), key=lambda n: -n.visits)
        children = list(map(lambda c: f"{c.action}: prior {c.prior} value_sum {float(c.value_sum)} visits {c.visits}", children))
        return f"Board: {self.board}\nNext player: {self.to_play}\nDerived policy: {self.derived_policy()}\n" + "\n".join(children)

class MCTS:
    def __init__(self, c_puct=4, dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def run(self, game: FiveInARow, model: Model, root: MCTS_Node, simulations = 80, temperature = 1.0):
        root.predict(model)
        root.expand(game)
        root.apply_dirichlet_noise(self.dirichlet_alpha, self.dirichlet_epsilon)
        for _ in range(simulations):
            # select
            node = root.select(self.c_puct)
            # set value and logits
            node.predict(model)
            # expand children
            node.expand(game)
            # eval and backprop
            node.eval_and_backprop()

        derived_policy = root.derived_policy(temperature)
        action = int(torch.multinomial(derived_policy,1).item())
        return root.children[action]

    def move_opponent(self, game, root: MCTS_Node, action):
        root.expand(game, ignore_priors=True) # make sure that the node is expanded
        return root.children[action]

def sample_agent_action(mcts, game, node, model, iters=80, temperature=1.0, iteration=0, display=False):
    best_node = mcts.run(game, model, node, iters, temperature)
    action = best_node.action
    if display:
        print(best_node.parent)
        if iteration > 0:
            print(f"iter {iteration} ",end='')
        else:
            print("BEST ",end='')
        print(f"MCTS agent: {action} {game.index_to_pos(action)}")
    return best_node

def sample_random_action(mcts, game, node, display=False):
    # equal probability
    logits = torch.zeros_like(game.board).flatten().where(game.board.flatten() == 0, float("-inf"))
    probs = F.softmax(logits,dim=0)
    action = torch.multinomial(probs,1).item()
    if display:
        print(f"Random agent: {action} {game.index_to_pos(action)}")
    next_node = mcts.move_opponent(game, node, action)
    return next_node

def sample_player_action(mcts, game, node):
    pos = input("position: ").split()
    while len(pos) != 2 or not game.can_place(game.board, int(pos[0]), int(pos[1])):
        print("input a valid position")
        pos = input("position: ").split()
    action = game.pos_to_index(int(pos[0]), int(pos[1]))
    next_node = mcts.move_opponent(game, node, action)
    return next_node

def set_lr(optim: optim.Optimizer, lr):
    for param_group in optim.param_groups:
        param_group['lr'] = lr

def update_lr_kl(optim: optim.Optimizer, lr, lr_mult, kl_value, kl_threshold = 0.02, kl_factor = 1.5, lr_factor = 1.5, min_lr_mult = 0.1, max_lr_mult = 10):
    # credit: junxiaosong/AlphaZero_Gomoku and https://skrl.readthedocs.io/en/latest/api/resources/schedulers/kl_adaptive.html
    if kl_value > kl_threshold * kl_factor and lr_mult > min_lr_mult:
        lr_mult /= lr_factor
    if kl_value < kl_threshold / kl_factor and lr_mult < max_lr_mult:
        lr_mult *= lr_factor

    set_lr(optim, lr * lr_mult)

    return lr, lr_mult

class Trainer():
    def __init__(self, model: Model, game: FiveInARow, c_puct=4, lr=2e-3, mcts_sim=80, epochs=20, batch_size=80, dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        self.model = model
        self.optim = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.lr = lr
        self.lr_mult = 1.0
        self.kl_threshold = 4e-5
        self.game = game
        self.mcts_sim = mcts_sim
        self.c_puct = c_puct
        self.epochs = epochs
        self.batch_size = batch_size
        self.memory = deque(maxlen=10000)
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def save(self, iteration, best=False):
        base_path = f"./models/five_in_a_row/iter{iteration}"
        os.makedirs(base_path, exist_ok=True)

        # Create symlink for best model if specified
        if best:
            best_path = "./models/five_in_a_row/best"
            if os.path.exists(best_path):
                os.remove(best_path)
            rel_path = os.path.relpath(base_path, os.path.dirname(best_path))
            os.symlink(rel_path, best_path)
            print(f"Updated best model symlink to iteration {iteration}")
            return

        # Save model state
        model_path = f"{base_path}/model.pth"
        torch.save(self.model.state_dict(), model_path)

        # Save metadata
        metadata = {
            'iteration': iteration,
            'optimizer_state': self.optim.state_dict(),
            'memory_size': len(self.memory),
            'hyperparameters': {
                'lr': self.lr,
                'lr_mult': self.lr_mult,
                'kl_threshold': self.kl_threshold,
                'c_puct': self.c_puct,
                'mcts_sim': self.mcts_sim,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
            }
        }
        metadata_path = f"{base_path}/metadata.pth"
        torch.save(metadata, metadata_path)

        print(f"Saved model and metadata for iteration {iteration}")

    def load(self, best, iteration=0):
        if best:
            base_path = "./models/five_in_a_row/best"
            if not os.path.exists(base_path):
                raise FileNotFoundError("No best model symlink found")
            base_path = os.path.realpath(base_path)  # Resolve the symlink
        else:
            base_path = f"./models/five_in_a_row/iter{iteration}"

        model_path = f"{base_path}/model.pth"
        metadata_path = f"{base_path}/metadata.pth"

        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"No saved model or metadata found for {'best' if best else f'iteration {iteration}'}")

        # Load model state
        self.model.load_state_dict(torch.load(model_path))

        # Load metadata
        metadata = torch.load(metadata_path)

        # Restore optimizer and scheduler states
        self.optim.load_state_dict(metadata['optimizer_state'])

        # Restore or update hyperparameters
        hp = metadata['hyperparameters']
        self.lr = hp['lr']
        self.lr_mult = hp['lr_mult']
        self.kl_threshold = hp['kl_threshold']
        self.c_puct = hp['c_puct']
        self.mcts_sim = hp['mcts_sim']
        self.epochs = hp['epochs']
        self.batch_size = hp['batch_size']

        loaded_iteration = metadata['iteration']
        print(f"Loaded {'best' if best else f'iteration {iteration}'} (actual iteration: {loaded_iteration})")

    def play_game(self):
        self.game.reset()
        game_start = MCTS_Node(0, self.game.board, self.game.player)
        mcts = MCTS(self.c_puct, self.dirichlet_alpha, self.dirichlet_epsilon)
        node = game_start
        game_memory = []

        winner = 0
        term = False
        while not term:
            temperature = 1.0 if self.game.turn < 30 else 0.1
            best_node = mcts.run(self.game, self.model, node, self.mcts_sim, temperature)
            game_memory.append((node.get_data(), node.to_play, node.derived_policy()))
            node = best_node
            del best_node.parent # not sure if this triggers GC but I want to keep memory low.
            best_node.parent = None
            _, winner, term, _, _ = self.game.step(best_node.action)

        print(f"Winner: {winner}", end = ' | ')
        for data, to_play, derived_policy in game_memory:
            self.memory.append((data, -to_play * winner, derived_policy))
            # print(f"Player: {to_play}\nBoard Canonical View\n{to_play * board}\nValue: {-to_play * winner}\nDerived Policy:\n{derived_policy.reshape(9,9)}")

        print(f"{len(game_memory)=}")

    def self_play(self, games=25):
        for game in range(games):
            print(f"Game {game+1}", end = ' | ')
            self.play_game()

    def train(self):
        all_state_data, _, _ = zip(*self.memory)
        all_state_data = torch.cat(all_state_data)

        old_predicted_values, old_predicted_policy_logits = self.model(all_state_data)
        old_log_probs = F.log_softmax(old_predicted_policy_logits, dim=0)

        for epoch in range(self.epochs):
            losses = []
            policy_losses = []
            value_losses = []

            for batch in batched(self.memory, self.batch_size):
                state_data, derived_values, derived_policies = zip(*batch)

                state_data = torch.cat(state_data)
                derived_policies = torch.stack(derived_policies)
                derived_values = tensor(derived_values).float().unsqueeze(dim=1)

                predicted_values, predicted_policy_logits = self.model(state_data)

                policy_loss = F.cross_entropy(predicted_policy_logits, derived_policies)
                value_loss = F.mse_loss(predicted_values, derived_values)

                loss = policy_loss + value_loss

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                losses.append(loss.item())
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

            avg_loss = np.mean(losses)
            avg_policy_loss = np.mean(policy_losses)
            avg_value_loss = np.mean(value_losses)
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Policy Loss: {avg_policy_loss:.4f} | Value Loss: {avg_value_loss:.4f}")

            new_predicted_values, new_predicted_policy_logits = self.model(all_state_data)
            new_log_probs = F.log_softmax(new_predicted_policy_logits, dim=0)

            kl_value = F.kl_div(input=new_log_probs, target=old_log_probs, log_target=True)
            print(f"{kl_value=}")

            if kl_value > self.kl_threshold * 4:  # early stopping if D_KL diverges badly (credit junxiaosong/AlphaZero_Gomoku)
                break

        self.lr, self.lr_mult = update_lr_kl(self.optim, self.lr, self.lr_mult, kl_value, kl_threshold=self.kl_threshold)
        print(f"Learning rate now {self.lr * self.lr_mult = }")

def eval_agent(args):
    mcts = MCTS(4, dirichlet_alpha=0.3, dirichlet_epsilon=0.25)
    game = FiveInARow()
    model = Model((game.rows,game.cols))
    model.load_state_dict(torch.load(f"./models/five_in_a_row/iter{args.iteration}/model.pth"))
    model.eval()

    opponent_model = Model((game.rows, game.cols))
    if args.opponent == "BEST":
        opponent_model.load_state_dict(torch.load(f"./models/five_in_a_row/best/model.pth"))

    print(f"Evaluating iteration {args.iteration} against {args.opponent}")

    num_games = 50
    total_wins = 0
    total_losses = 0
    total_ties = 0

    temperature = 0.1

    for agent_color in [WHITE, BLACK]:
        wins = 0
        losses = 0
        ties = 0
        winners = []
        for _ in range(num_games//2):
            game.reset()
            agent_node = MCTS_Node(0, game.board, game.player)
            opponent_node = MCTS_Node(0, game.board, game.player)
            term = False
            while not term:
                if game.player == agent_color:
                    agent_node = sample_agent_action(mcts, game, agent_node, model, temperature=temperature, iteration=args.iteration, display=False)
                    action = agent_node.action
                    opponent_node = mcts.move_opponent(game, opponent_node, action)
                elif args.opponent == "BEST":
                    opponent_node = sample_agent_action(mcts, game, opponent_node, opponent_model, temperature=temperature, display=False)
                    action = opponent_node.action
                    agent_node = mcts.move_opponent(game, agent_node, action)
                else:
                    agent_node = sample_random_action(mcts, game, agent_node)
                    action = agent_node.action

                _, _, term, _, _ = game.step(action)

            print(game.board)
            winner = game.get_winner(game.board)
            print(f"Winner: {winner}")
            winners.append(winner)
            wins += (winner == agent_color)
            losses += (winner == -agent_color)
            ties += (winner == 0)
        print(f"Agent = {"WHITE" if agent_color == WHITE else "BLACK"} (Wins/Losses/Ties): {wins}/{losses}/{ties}")
        total_wins += wins
        total_losses += losses
        total_ties += ties

    print(f"Agent (Wins/Losses/Ties): {total_wins}/{total_losses}/{total_ties}")

    is_agent_better = (total_wins + total_ties / 2) / num_games > 0.55
    return is_agent_better

def train(args):
    game = FiveInARow()
    model = Model((game.rows,game.cols))
    trainer = Trainer(model, game, lr=1e-4, c_puct=4, mcts_sim=80, epochs=20, dirichlet_alpha=0.3, dirichlet_epsilon=0.25)
    if args.iteration != 0:
        trainer.load(best=False, iteration=args.iteration)
    model.train()

    training_iterations = 200
    games_per_iteration = 25

    for iter in range(args.iteration+1, training_iterations+1):
        print(f"iteration {iter}/{training_iterations}")
        trainer.self_play(games_per_iteration)
        trainer.train()
        trainer.save(iter)

        if iter % 5 == 0:
            best_path = "./models/five_in_a_row/best"
            args.iteration = iter
            if not os.path.exists(best_path):
                args.opponent = "RANDOM"
            else:
                args.opponent = "BEST"
            is_better = eval_agent(args)
            trainer.save(iter, best=is_better)

def play(args):
    if args.player == "WHITE":
        player_color = WHITE
    else:
        player_color = BLACK

    game = FiveInARow()
    mcts = MCTS(4, dirichlet_alpha=0.3, dirichlet_epsilon=0.25)
    model = Model((game.rows,game.cols))
    model.load_state_dict(torch.load(f"./models/five_in_a_row/best/model.pth"))
    model.eval()

    game.reset()
    game_start = MCTS_Node(0, game.board, game.player)
    node = game_start
    print(game.board)
    term = False
    while not term:
        if game.player == player_color:
            node = sample_player_action(mcts, game, node)
            action = node.action
        else:
            temperature = 0.1  # Lower temperature for stronger play
            node = sample_agent_action(mcts, game, node, model, temperature=temperature, display=True)
            action = node.action

        _, _, term, _, _ = game.step(action)
        print(game.board)

    print(f"Winner: {game.get_winner(game.board)}")

import argparse
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(required=True)
parser_train = subparsers.add_parser("train")
parser_train.add_argument('--iteration', default=0, required=False, type=int, action="store")
parser_train.add_argument('--opponent', choices=["BEST","RANDOM"], default="BEST", required=False)
parser_train.set_defaults(func=train)
parser_play = subparsers.add_parser("play")
parser_play.add_argument('--player', choices=["WHITE","BLACK"], default="WHITE", required=False)
parser_play.set_defaults(func=play)
parser_eval = subparsers.add_parser("eval")
parser_eval.add_argument('--iteration', default=0, required=False, type=int, action="store")
parser_eval.add_argument('--opponent', choices=["BEST","RANDOM"], default="BEST", required=False)
parser_eval.set_defaults(func=eval_agent)

if __name__ == "__main__":
    # torch.set_anomaly_enabled(True)
    args = parser.parse_args()
    args.func(args)
