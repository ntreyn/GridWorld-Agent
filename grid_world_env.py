import random


class grid_env:

    reward_map = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 20,
        7: 0,
        8: 0
    }

    def __init__(self):
        self.state_space = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.action_space = [0, 1, 2, 3]
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
        self.position = (0, 0)
        self.done = False
        self.grid[self.position[0]][self.position[1]] = 'A'
        return self.get_state()

    def step(self, action):

        reward = -1
        new_row, new_col = self.position

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

        print(self.position)

        new_state = self.get_state()
        reward += self.reward_map[new_state]

        return new_state, reward, self.done

    def render(self):
        print('-------------')
        for row in range(3):
            rowString = '| ' + self.grid[row][0] + ' | ' + self.grid[row][1] + ' | ' + self.grid[row][2] + ' |'
            print(rowString)
            print('-------------')

    def sample_action(self):
        return random.choice(self.potential_moves())

    def potential_moves(self):
        r, c = self.position
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
    
    def get_state(self):
        return self.position_to_state(self.position)

    def position_to_state(self, pos):
        r, c = pos
        state = r * 3 + c
        return state

