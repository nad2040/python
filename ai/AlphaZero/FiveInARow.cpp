#include <torch/torch.h>
#include <algorithm>
#include <cmath>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

using Tensor = torch::Tensor;

constexpr int WHITE = 1;
constexpr int BLACK = -1;

class FiveInARow {
public:
    int rows, cols;
    int turn;

    FiveInARow(int rows = 9, int cols = 9, int player = WHITE)
        : rows(rows), cols(cols), player(player), turn(0), winner(0) {
        board = torch::zeros({rows, cols}, torch::kInt32);
    }

    inline int pos_to_index(int row, int col) const { return row * cols + col; }

    inline std::pair<int, int> index_to_pos(int index) const {
        return {index / cols, index % cols};
    }

    bool can_place(const Tensor& board, int row, int col) const {
        return row >= 0 && row < rows && col >= 0 && col < cols &&
               board[row][col].item<int>() == 0;
    }

    Tensor place(const Tensor& board, int row, int col, int player) const {
        Tensor new_board = board.clone();
        new_board[row][col] = player;
        return new_board;
    }

    std::vector<std::pair<int, Tensor>> next_states(const Tensor& board,
                                                    int to_play) const {
        std::vector<std::pair<int, Tensor>> states;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (can_place(board, r, c)) {
                    states.push_back(
                        {pos_to_index(r, c), place(board, r, c, to_play)});
                }
            }
        }
        return states;
    }

    int count_pieces(const Tensor& board, int row, int col) const {
        std::vector<std::vector<std::pair<int, int>>> directions = {
            {{-1, 0}, {1, 0}},
            {{0, -1}, {0, 1}},
            {{-1, 1}, {1, -1}},
            {{1, 1}, {-1, -1}}};
        int piece = board[row][col].item<int>();
        int max_count = 0;

        for (const auto& dir : directions) {
            int count = 1;
            for (const auto& d : dir) {
                int dr = d.first, dc = d.second;
                int i = 1;
                while (row + i * dr >= 0 && row + i * dr < rows &&
                       col + i * dc >= 0 && col + i * dc < cols &&
                       board[row + i * dr][col + i * dc].item<int>() == piece) {
                    count += 1;
                    i += 1;
                }
            }
            max_count = std::max(max_count, count);
        }
        return max_count;
    }

    bool check_win(const Tensor& board, int row, int col) const {
        return count_pieces(board, row, col) >= 5;
    }

    int get_winner(const Tensor& board) const {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (board[r][c].item<int>() != 0 && check_win(board, r, c)) {
                    return board[r][c].item<int>();
                }
            }
        }
        return 0;
    }

    bool is_game_over(const Tensor& board) const {
        return get_winner(board) != 0 || torch::all(board != 0).item<bool>();
    }

    void reset() {
        board = torch::zeros({rows, cols}, torch::kInt32);
        player = WHITE;
        winner = 0;
    }

    std::tuple<Tensor, int, bool> step(int action) {
        auto [row, col] = index_to_pos(action);
        board = place(board, row, col, player);
        bool term = is_game_over(board);
        winner = get_winner(board);
        int reward = winner;
        player = -player;
        return {board, reward, term};
    }

    Tensor get_board() const { return board; }
    int get_player() const { return player; }
    int get_winner_val() const { return winner; }

private:
    int player;
    int winner;
    Tensor board;
};

struct Model : torch::nn::Module {
    torch::nn::Sequential cnn;
    torch::nn::Sequential value_head;
    torch::nn::Sequential policy_head;

    int rows, cols;

    Model(std::pair<int, int> game_dim) {
        rows = game_dim.first;
        cols = game_dim.second;

        // CNN feature extractor
        cnn = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(4, 32, 3).padding(1)),
            torch::nn::BatchNorm2d(32), torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)),
            torch::nn::BatchNorm2d(64), torch::nn::ReLU(),
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(64, 128, 3)),  // no padding
            torch::nn::BatchNorm2d(128), torch::nn::ReLU(),
            torch::nn::Flatten(), torch::nn::Dropout(0.3));
        register_module("cnn", cnn);

        int linear_input = 128 * (rows - 2) * (cols - 2);  // matches Python

        // Value head
        value_head = torch::nn::Sequential(
            torch::nn::Linear(linear_input, 256), torch::nn::ReLU(),
            torch::nn::Linear(256, 1), torch::nn::Tanh());
        register_module("value_head", value_head);

        // Policy head
        policy_head = torch::nn::Sequential(
            torch::nn::Linear(linear_input, 256), torch::nn::ReLU(),
            torch::nn::Linear(256, rows * cols));
        register_module("policy_head", policy_head);
    }

    std::pair<Tensor, Tensor> forward(Tensor data) {
        auto x = cnn->forward(data);
        auto value = value_head->forward(x);
        auto logits = policy_head->forward(x);
        return {value, logits};
    }
};

class MCTS_Node : public std::enable_shared_from_this<MCTS_Node> {
public:
    float prior;
    Tensor board;
    int to_play;
    std::shared_ptr<MCTS_Node> parent;
    int action;
    Tensor predicted_value;
    Tensor predicted_priors;
    std::map<int, std::shared_ptr<MCTS_Node>> children;
    int visits;
    float value_sum;

    MCTS_Node(float prior, const Tensor& board, int to_play,
              std::shared_ptr<MCTS_Node> parent = nullptr, int action = -1)
        : prior(prior),
          board(board),
          to_play(to_play),
          parent(parent),
          action(action),
          visits(0),
          value_sum(0.0) {}

    float value() const { return visits > 0 ? value_sum / visits : 0.0f; }

    float uct_score(float c_puct) const {
        if (!parent)
            throw std::runtime_error("Parent is null for UCT");
        float Q = value();
        float U = prior * c_puct * std::sqrt(parent->visits) / (1 + visits);
        return Q + U;
    }

    Tensor get_data() const {
        auto shape = board.sizes();
        int rows = shape[0];
        int cols = shape[1];
        Tensor data = torch::zeros({1, 4, rows, cols});
        data[0][0] = (board == to_play).to(torch::kFloat32);
        data[0][1] = (board == -to_play).to(torch::kFloat32);
        if (action >= 0) {
            data[0][2][action / cols][action % cols] = 1.0;
        }
        data[0][3] = to_play;
        return data;
    }

    bool expanded() const { return !children.empty(); }

    Tensor derived_policy(float temperature = 1.0f) const {
        if (!predicted_priors.defined())
            throw std::runtime_error("Predicted priors undefined");
        Tensor probs = torch::zeros_like(predicted_priors);
        for (const auto& [act, child] : children) {
            probs[act] = static_cast<float>(child->visits) / visits;
        }
        Tensor logits = torch::log(probs + 1e-10);  // avoid log(0)
        return torch::softmax(logits / temperature, 0);
    }

    template <typename Model>
    void predict(Model& model) {
        if (predicted_value.defined() && predicted_priors.defined())
            return;
        torch::NoGradGuard no_grad;
        auto input = get_data();
        input = input.to(torch::kFloat32);
        auto [pred_value, pred_logits] = model.forward(input);
        predicted_value = pred_value.flatten();
        Tensor mask = (board.flatten() == 0).to(torch::kFloat32);
        Tensor masked_logits =
            pred_logits.flatten() * mask + (-1e9) * (1 - mask);
        predicted_priors = torch::softmax(masked_logits, 0);
    }

    void apply_dirichlet_noise(float alpha, float epsilon) {
        if (!expanded())
            throw std::runtime_error("Cannot apply noise on unexpanded node");
        std::vector<int> legal_actions;
        for (const auto& [act, _] : children)
            legal_actions.push_back(act);

        std::vector<double> noise(legal_actions.size());
        std::random_device rd;
        std::mt19937 gen(rd());
        std::gamma_distribution<> gamma(alpha, 1.0);

        for (size_t i = 0; i < noise.size(); ++i)
            noise[i] = gamma(gen);
        double sum = std::accumulate(noise.begin(), noise.end(), 0.0);
        for (size_t i = 0; i < noise.size(); ++i)
            noise[i] /= sum;

        for (size_t i = 0; i < legal_actions.size(); ++i) {
            int act = legal_actions[i];
            children[act]->prior =
                (1 - epsilon) * children[act]->prior + epsilon * noise[i];
        }
    }

    std::shared_ptr<MCTS_Node> select(float c_puct) {
        auto node = shared_from_this();
        while (node->expanded()) {
            auto best_it =
                std::max_element(node->children.begin(), node->children.end(),
                                 [c_puct](const auto& a, const auto& b) {
                                     return a.second->uct_score(c_puct) <
                                            b.second->uct_score(c_puct);
                                 });
            node = best_it->second;
        }
        return node;
    }

    void expand(FiveInARow& game, bool ignore_priors = false) {
        if (!expanded()) {
            auto next_states = game.next_states(board, to_play);
            for (auto& [act, new_board] : next_states) {
                float prior_val =
                    ignore_priors ? 0.0f : predicted_priors[act].item<float>();
                children[act] = std::make_shared<MCTS_Node>(
                    prior_val, new_board, -to_play, shared_from_this(), act);
            }
        }
    }

    void eval_and_backprop() {
        if (!predicted_value.defined())
            throw std::runtime_error("Predicted value undefined");
        float value = predicted_value.item<float>();
        auto node = shared_from_this();
        while (node) {
            node->value_sum += value;
            node->visits += 1;
            node = node->parent;
            value = -value;
        }
    }
};

class MCTS {
public:
    float c_puct;
    float dirichlet_alpha;
    float dirichlet_epsilon;

    MCTS(float c_puct = 5.0, float alpha = 0.3, float epsilon = 0.25)
        : c_puct(c_puct), dirichlet_alpha(alpha), dirichlet_epsilon(epsilon) {}

    template <typename Model>
    std::shared_ptr<MCTS_Node> run(FiveInARow& game, Model& model,
                                   std::shared_ptr<MCTS_Node> root,
                                   int simulations = 200,
                                   float temperature = 1.0f) {
        root->predict(model);
        root->expand(game);
        root->apply_dirichlet_noise(dirichlet_alpha, dirichlet_epsilon);

        for (int i = 0; i < simulations; ++i) {
            auto node = root->select(c_puct);
            node->predict(model);
            node->expand(game);
            node->eval_and_backprop();
        }

        Tensor policy = root->derived_policy(temperature);
        int action = torch::multinomial(policy, 1).item<int>();
        return root->children[action];
    }

    std::shared_ptr<MCTS_Node> move_opponent(FiveInARow& game,
                                             std::shared_ptr<MCTS_Node> root,
                                             int action) {
        root->expand(game, true);  // ensure the node is expanded
        return root->children[action];
    }
};

// --- Sample MCTS agent action ---
template <typename Model>
std::shared_ptr<MCTS_Node> sample_agent_action(MCTS& mcts, FiveInARow& game,
                                               std::shared_ptr<MCTS_Node> node,
                                               Model& model, int iters = 200,
                                               float temperature = 1.0,
                                               int iteration = 0,
                                               bool display = false) {
    auto best_node = mcts.run(game, model, node, iters, temperature);
    int action = best_node->action;

    if (display && best_node->parent) {
        std::cout << "Derived policy and children:\n"
                  << "Parent node data not printed here (add custom print if "
                     "needed)\n";

        if (iteration > 0)
            std::cout << "iter " << iteration << " ";
        else
            std::cout << "BEST ";
        auto [row, col] = game.index_to_pos(action);
        std::cout << "MCTS agent: " << action << " (" << row << "," << col
                  << ")\n";
    }

    return best_node;
}

// --- Sample random action ---
std::shared_ptr<MCTS_Node> sample_random_action(MCTS& mcts, FiveInARow& game,
                                                std::shared_ptr<MCTS_Node> node,
                                                bool display = false) {
    Tensor logits =
        torch::zeros_like(game.get_board(), torch::kFloat32).flatten();
    Tensor mask = (game.get_board().flatten() == 0).to(torch::kFloat32);
    logits = logits.masked_fill(mask == 0, -1e9f);
    Tensor probs = torch::softmax(logits, 0);
    int action = torch::multinomial(probs, 1).item<int>();

    if (display) {
        auto [row, col] = game.index_to_pos(action);
        std::cout << "Random agent: " << action << " (" << row << "," << col
                  << ")\n";
    }

    auto next_node = mcts.move_opponent(game, node, action);
    return next_node;
}

// --- Sample human player action ---
std::shared_ptr<MCTS_Node> sample_player_action(
    MCTS& mcts, FiveInARow& game, std::shared_ptr<MCTS_Node> node) {
    int row = -1, col = -1;
    while (true) {
        std::cout << "Position (row col): ";
        std::string line;
        std::getline(std::cin, line);
        std::istringstream iss(line);
        if (iss >> row >> col) {
            if (game.can_place(game.get_board(), row, col))
                break;
        }
        std::cout << "Input a valid position.\n";
    }

    int action = game.pos_to_index(row, col);
    auto next_node = mcts.move_opponent(game, node, action);
    return next_node;
}

// --- Set learning rate ---
void set_lr(torch::optim::Optimizer& optim, float lr) {
    for (auto& group : optim.param_groups()) {
        group.options().set_lr(lr);
    }
}

// --- Update learning rate using KL ---
float update_lr_kl(torch::optim::Optimizer& optim, float lr, float& lr_mult,
                   float kl_value, float kl_threshold = 0.02,
                   float kl_factor = 1.5, float lr_factor = 1.5,
                   float min_lr_mult = 0.1, float max_lr_mult = 10.0) {
    if (kl_value > kl_threshold * kl_factor && lr_mult > min_lr_mult) {
        lr_mult /= lr_factor;
    }
    if (kl_value < kl_threshold / kl_factor && lr_mult < max_lr_mult) {
        lr_mult *= lr_factor;
    }

    set_lr(optim, lr * lr_mult);
    return lr;
}

namespace fs = std::filesystem;

struct MemoryItem {
    torch::Tensor state;
    float value;
    torch::Tensor policy;
};

struct TrainerMetadata {
    int iteration;
    float lr;
    int mcts_sim;
    int batch_size;
    float dirichlet_alpha;
    float dirichlet_epsilon;
};

class Trainer {
public:
    Trainer(std::shared_ptr<Model> model, FiveInARow& game, float c_puct = 5.0f,
            float lr = 2e-3, int mcts_sim = 200, int epochs = 20,
            int batch_size = 80, float dirichlet_alpha = 0.3f,
            float dirichlet_epsilon = 0.25f)
        : model(model),
          game(game),
          c_puct(c_puct),
          lr(lr),
          lr_mult(1.0f),
          kl_threshold(4e-5),
          mcts_sim(mcts_sim),
          epochs(epochs),
          batch_size(batch_size),
          dirichlet_alpha(dirichlet_alpha),
          dirichlet_epsilon(dirichlet_epsilon) {
        optim = std::make_unique<torch::optim::Adam>(
            model->parameters(),
            torch::optim::AdamOptions(lr).weight_decay(1e-4));
    }

    void save(int iteration, bool best = false) {
        fs::path base_path =
            "./models/five_in_a_row/iter" + std::to_string(iteration);
        fs::create_directories(base_path);

        if (best) {
            fs::path best_path = "./models/five_in_a_row/best";
            if (fs::exists(best_path))
                fs::remove(best_path);

            fs::path rel_path =
                fs::relative(base_path, best_path.parent_path());
            fs::create_symlink(rel_path, best_path);

            std::cout << "Updated best model symlink to iteration " << iteration
                      << "\n";
            return;
        }

        torch::save(model, base_path / "model.pt");

        std::map<std::string, torch::Tensor> metadata;
        metadata["iteration"] = torch::tensor(iteration);
        auto [states, policies, values] = memory_to_tensors();
        metadata["memory_states"] = states;
        metadata["memory_policies"] = policies;
        metadata["memory_values"] = values;
        metadata["memory_size"] =
            torch::tensor(static_cast<int>(memory.size()));
        metadata["lr"] = torch::tensor(lr);
        metadata["lr_mult"] = torch::tensor(lr_mult);
        metadata["kl_threshold"] = torch::tensor(kl_threshold);
        metadata["c_puct"] = torch::tensor(c_puct);
        metadata["mcts_sim"] = torch::tensor(mcts_sim);
        metadata["epochs"] = torch::tensor(epochs);
        metadata["batch_size"] = torch::tensor(batch_size);

        torch::serialize::OutputArchive metadata_archive;
        for (const auto& pair : metadata) {
            metadata_archive.write(pair.first, pair.second);
        }
        metadata_archive.save_to(base_path / "metadata.pt");

        // Save optimizer state
        torch::serialize::OutputArchive optim_archive;
        optim->save(optim_archive);
        optim_archive.save_to(base_path / "optimizer.pt");

        std::cout << "Saved model, metadata, and optimizer for iteration "
                  << iteration << "\n";
    }

    void load(bool best, int iteration = 0) {
        fs::path base_path;
        if (best) {
            base_path = "./models/five_in_a_row/best";
            if (!fs::exists(base_path))
                throw std::runtime_error("No best model symlink found");
            base_path = fs::canonical(base_path);
        } else {
            base_path =
                "./models/five_in_a_row/iter" + std::to_string(iteration);
        }

        // Load model
        torch::load(model, base_path / "model.pt");

        // Load metadata
        std::map<std::string, torch::Tensor> metadata;
        torch::serialize::InputArchive metadata_archive;
        metadata_archive.load_from(base_path / "metadata.pt");

        for (auto& pair : metadata)
            pair.second = torch::Tensor();  // initialize
        for (const auto& key :
             {"iteration", "memory_states", "memory_policies", "memory_values",
              "memory_size", "lr", "lr_mult", "kl_threshold", "c_puct",
              "mcts_sim", "epochs", "batch_size"}) {
            torch::Tensor tmp;
            metadata_archive.read(key, tmp);
            metadata[key] = tmp;
        }

        // Restore memory
        auto states = metadata.at("memory_states");
        auto policies = metadata.at("memory_policies");
        auto values = metadata.at("memory_values");
        memory_from_tensors(states, policies, values);

        // Load optimizer
        torch::serialize::InputArchive optim_archive;
        optim_archive.load_from(base_path / "optimizer.pt");
        optim->load(optim_archive);

        // Restore hyperparameters
        lr = metadata.at("lr").item<float>();
        lr_mult = metadata.at("lr_mult").item<float>();
        kl_threshold = metadata.at("kl_threshold").item<float>();
        c_puct = metadata.at("c_puct").item<float>();
        mcts_sim = metadata.at("mcts_sim").item<int>();
        epochs = metadata.at("epochs").item<int>();
        batch_size = metadata.at("batch_size").item<int>();

        int loaded_iteration = metadata.at("iteration").item<int>();

        std::cout << "Loaded "
                  << (best ? "best"
                           : ("iteration " + std::to_string(iteration)))
                  << " (actual iteration: " << loaded_iteration << ")\n";
    }

    void play_game() {
        game.reset();
        auto game_start =
            std::make_shared<MCTS_Node>(0, game.get_board(), game.get_player());
        MCTS mcts(c_puct, dirichlet_alpha, dirichlet_epsilon);
        auto node = game_start;

        std::vector<MemoryItem> game_memory;
        bool term = false;
        int winner = 0;

        while (!term) {
            float temperature = (game.turn < 30) ? 1.0f : 0.1f;
            auto best_node =
                mcts.run(game, *model, node, mcts_sim, temperature);
            MemoryItem item{node->get_data(), (float)-node->to_play * winner,
                            node->derived_policy()};
            game_memory.push_back(item);

            node = best_node;
            best_node->parent.reset();  // release parent pointer
            best_node->parent = nullptr;

            auto [board, reward, done] = game.step(best_node->action);
            winner = game.get_winner(board);
            term = done;
        }

        std::cout << "Winner: " << winner
                  << " | Memory collected: " << game_memory.size() << "\n";

        for (auto& item : game_memory) {
            memory.push_back(item);
            if (memory.size() > 10000)
                memory.pop_front();
        }
    }

    void self_play(int games = 25) {
        for (int i = 0; i < games; ++i) {
            std::cout << "Game " << i + 1 << " | ";
            play_game();
        }
    }

    std::tuple<Tensor, Tensor, Tensor> memory_to_tensors() {
        // Convert memory to tensors
        std::vector<torch::Tensor> states, policies;
        std::vector<float> values;
        for (auto& item : memory) {
            states.push_back(item.state);
            policies.push_back(item.policy);
            values.push_back(item.value);
        }
        auto state_tensor = torch::cat(states);
        auto policy_tensor = torch::stack(policies);
        auto value_tensor = torch::tensor(values).unsqueeze(1);
        return std::make_tuple(state_tensor, policy_tensor, value_tensor);
    }

    void memory_from_tensors(const torch::Tensor& state_tensor,
                             const torch::Tensor& policy_tensor,
                             const torch::Tensor& value_tensor) {
        memory.clear();

        int64_t N = state_tensor.size(0);  // number of samples
        for (int64_t i = 0; i < N; ++i) {
            MemoryItem item;
            item.state =
                state_tensor[i].unsqueeze(0);  // keep batch dim consistent
            item.policy = policy_tensor[i];
            item.value = value_tensor[i].item<float>();
            memory.push_back(item);
        }
    }

    void train() {
        auto [state_tensor, policy_tensor, value_tensor] = memory_to_tensors();

        for (int epoch = 0; epoch < epochs; ++epoch) {
            optim->zero_grad();
            auto [pred_values, pred_policies] = model->forward(state_tensor);
            auto policy_loss = torch::nn::functional::cross_entropy(
                pred_policies, policy_tensor);
            auto value_loss = torch::mse_loss(pred_values, value_tensor);
            auto loss = policy_loss + value_loss;
            loss.backward();
            optim->step();

            std::cout << "Epoch " << epoch + 1
                      << " | Loss: " << loss.item<float>()
                      << " | Policy: " << policy_loss.item<float>()
                      << " | Value: " << value_loss.item<float>() << "\n";
        }
    }

private:
    std::shared_ptr<Model> model;
    FiveInARow& game;
    std::unique_ptr<torch::optim::Adam> optim;
    float lr;
    float lr_mult;
    float kl_threshold;
    float c_puct;
    int mcts_sim;
    int epochs;
    int batch_size;
    float dirichlet_alpha;
    float dirichlet_epsilon;
    std::deque<MemoryItem> memory;
};

// ----- Evaluate the agent -----
bool eval_agent(int iteration, const std::string& opponent = "BEST",
                int num_games = 50) {
    FiveInARow game;
    auto model = std::make_shared<Model>(std::make_pair(game.rows, game.cols));
    torch::load(model, "./models/five_in_a_row/iter" +
                           std::to_string(iteration) + "/model.pt");
    model->eval();

    std::shared_ptr<Model> opponent_model =
        std::make_shared<Model>(std::make_pair(game.rows, game.cols));
    if (opponent == "BEST") {
        torch::load(opponent_model, "./models/five_in_a_row/best/model.pt");
        opponent_model->eval();
    }

    MCTS mcts(4, 0.3f, 0.25f);

    int total_wins = 0, total_losses = 0, total_ties = 0;

    for (int agent_color : {WHITE, BLACK}) {
        int wins = 0, losses = 0, ties = 0;

        for (int g = 0; g < num_games / 2; ++g) {
            game.reset();
            auto agent_node = std::make_shared<MCTS_Node>(0, game.get_board(),
                                                          game.get_player());
            auto opponent_node = std::make_shared<MCTS_Node>(
                0, game.get_board(), game.get_player());
            bool term = false;

            while (!term) {
                int action = -1;
                if (game.get_player() == agent_color) {
                    agent_node = sample_agent_action(
                        mcts, game, agent_node, *model, 200, 0.1f, 0, false);
                    action = agent_node->action;
                    opponent_node =
                        mcts.move_opponent(game, opponent_node, action);
                } else if (opponent == "BEST") {
                    opponent_node = sample_agent_action(
                        mcts, game, opponent_node, *opponent_model, 200, 0.1f,
                        0, false);
                    action = opponent_node->action;
                    agent_node = mcts.move_opponent(game, agent_node, action);
                } else {
                    agent_node =
                        sample_random_action(mcts, game, agent_node, false);
                    action = agent_node->action;
                }

                auto [board, reward, done] = game.step(action);
                term = done;
            }

            int winner = game.get_winner(game.get_board());
            wins += (winner == agent_color);
            losses += (winner == -agent_color);
            ties += (winner == 0);
        }

        std::cout << "Agent = " << ((agent_color == WHITE) ? "WHITE" : "BLACK")
                  << " | Wins/Losses/Ties: " << wins << "/" << losses << "/"
                  << ties << "\n";

        total_wins += wins;
        total_losses += losses;
        total_ties += ties;
    }

    std::cout << "Total (Wins/Losses/Ties): " << total_wins << "/"
              << total_losses << "/" << total_ties << "\n";
    bool is_agent_better = (total_wins + total_ties / 2.0) / num_games > 0.55;
    return is_agent_better;
}

// ----- Interactive play -----
void play_game_interactive(const std::string& player_color_str) {
    int player_color = (player_color_str == "WHITE") ? WHITE : BLACK;
    FiveInARow game;
    auto model = std::make_shared<Model>(std::make_pair(game.rows, game.cols));
    torch::load(model, "./models/five_in_a_row/best/model.pt");
    model->eval();

    MCTS mcts(4, 0.3f, 0.25f);
    game.reset();
    auto node =
        std::make_shared<MCTS_Node>(0, game.get_board(), game.get_player());

    bool term = false;
    while (!term) {
        int action = -1;
        if (game.get_player() == player_color) {
            node = sample_player_action(mcts, game, node);
            action = node->action;
        } else {
            node = sample_agent_action(mcts, game, node, *model, 200, 0.1f, 0,
                                       true);
            action = node->action;
        }

        auto [board, reward, done] = game.step(action);
        term = done;
        std::cout << board << "\n";
    }

    std::cout << "Winner: " << game.get_winner(game.get_board()) << "\n";
}

// ----- Train function -----
void train(int start_iteration = 0, int training_iterations = 200,
           int games_per_iteration = 25) {
    FiveInARow game;
    auto model = std::make_shared<Model>(std::make_pair(game.rows, game.cols));
    Trainer trainer(model, game, 5.0f, 1e-4f, 200, 20, 80, 0.3f, 0.25f);

    if (start_iteration != 0) {
        trainer.load(false, start_iteration);
    }

    model->train();

    for (int iter = start_iteration + 1; iter <= training_iterations; ++iter) {
        std::cout << "Iteration " << iter << "/" << training_iterations << "\n";

        // Self-play to generate training data
        trainer.self_play(games_per_iteration);

        // Train model on collected memory
        trainer.train();

        // Save model
        trainer.save(iter);

        // Periodic evaluation every 5 iterations
        if (iter % 5 == 0) {
            std::string best_path = "./models/five_in_a_row/best";
            std::string opponent =
                std::filesystem::exists(best_path) ? "BEST" : "RANDOM";

            bool is_better = eval_agent(iter, opponent);

            // Update best model if new agent is stronger
            trainer.save(iter, is_better);
        }
    }
}

// ----- Example main() CLI -----
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " [train|play|eval] [options]\n";
        return 0;
    }

    std::string mode = argv[1];

    if (mode == "train") {
        int iteration = 0;
        if (argc >= 3)
            iteration = std::stoi(argv[2]);
        train(iteration);
    } else if (mode == "play") {
        std::string player_color = "WHITE";
        if (argc >= 3)
            player_color = argv[2];
        play_game_interactive(player_color);
    } else if (mode == "eval") {
        int iteration = 0;
        std::string opponent = "BEST";
        if (argc >= 3)
            iteration = std::stoi(argv[2]);
        if (argc >= 4)
            opponent = argv[3];
        eval_agent(iteration, opponent);
    } else {
        std::cerr << "Unknown mode: " << mode << "\n";
        return 1;
    }

    return 0;
}
