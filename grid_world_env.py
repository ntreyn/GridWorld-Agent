import random


class grid_env:

    reward_map = {
        0: 0,
        1: 0,
        2: 10,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 5,
        8: 0
    }
    visited_reward_map = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0
    }

    def __init__(self):
        self.action_space = [0, 1, 2, 3]
        self.state_size = (2 ** 9) * 9
        self.action_size = 4
        self.generate_states()
        """
        0: left
        1: right
        2: up
        3: down
        """
        self.reset()

    def reset(self):
        self.grid = [
            [ ' ', ' ', ' ' ],
            [ ' ', ' ', ' ' ],
            [ 'G', ' ', ' ' ]
        ]
        self.visited = [1, 0, 0, 0, 0, 0, 0, 0, 0]
        self.position = (0, 0)
        self.done = False
        self.grid[self.position[0]][self.position[1]] = 'A'
        return self.get_state()

    def step(self, action):

        reward = -1
        new_row, new_col = self.position

        legal_actions = self.potential_moves(self.position)
        if action not in legal_actions:
            new_state = self.get_state()
            reward = -1000
            return new_state, reward, self.done

        if action == 0:
            new_col += -1
        elif action == 1:
            new_col += 1
        elif action == 2:
            new_row += -1
        elif action == 3:
            new_row += 1

        if self.grid[new_row][new_col] == 'G':
            self.grid[new_row][new_col] = '$'
            self.done = True
        else:
            self.grid[new_row][new_col] = 'A'

        self.grid[self.position[0]][self.position[1]] = ' '
        self.position = (new_row, new_col)

        new_state = self.get_state()
        reward += self.get_reward(new_state)

        return new_state, reward, self.done

    def render(self):
        print('-------------')
        for row in range(3):
            rowString = '| ' + self.grid[row][0] + ' | ' + self.grid[row][1] + ' | ' + self.grid[row][2] + ' |'
            print(rowString)
            print('-------------')

    def sample_action(self):
        return random.choice(self.potential_moves(self.position))

    def get_reward(self, state):
        pos = self.state_to_pos(state)
        tile = self.pos_to_index(pos)
        if self.visited[tile] == 0:
            reward = self.reward_map[tile]
            self.visited[tile] = 1
        else:
            reward = self.visited_reward_map[tile]
        return reward

    def potential_moves(self, pos):
        r, c = pos
        moves = []
        
        if c - 1 >= 0:
            moves.append(0)
        if c + 1 <= 2:
            moves.append(1)
        if r - 1 >= 0:
            moves.append(2)
        if r + 1 <= 2:
            moves.append(3)

        return moves

    def bound_qtable(self, state, action):
        pos = self.state_to_pos(state)
        moves = self.potential_moves(pos)
        
        if action in moves:
            return False
        else:
            return True
    
    def get_state(self):
        return self.position_to_state(self.position)

    def pos_to_index(self, pos):
        r, c = pos
        index = r * 3 + c
        return index

    def position_to_state(self, pos):
        tile = self.pos_to_index(pos)
        board_state = tuple(self.visited)
        state = (board_state, tile)
        return self.state_space[state]
    
    def state_to_pos(self, state):
        _, tile = self.reverse_state_space[state]

        c = tile % 3
        r = tile // 3
        return (r, c)

    def generate_states(self):
        state = [0] * 9
        self.state_space = {}
        self.state_count = 0
        self.recursive_states(state, 0)
        self.reverse_state_space = {v: k for k, v in self.state_space.items()}

    def recursive_states(self, list, ind):
        if ind == 9:
            board_state = tuple(list)
            for tile in range(9):
                state = (board_state, tile)
                self.state_space[state] = self.state_count
                self.state_count += 1
            return
        else:
            l0 = list
            l0[ind] = 0
            self.recursive_states(list, ind + 1)
            l1 = list
            l1[ind] = 1
            self.recursive_states(l1, ind + 1)
            return

