import random
from tqdm import tqdm  # Progress bar


def get_all_actions(state):
    '''
    Input: (str) state
    Output: All actions one can choose from the state
    '''

    actions = []
    for col, s in enumerate(list(state)):
        if int(s) != 0:
            n = int(s)
            while n != 0:
                actions.append(str(col) + str(n))
                n = n - 1

    return actions


def result(s, a):
    '''
    Input:
        s: (str) State, e.g. '012'
        a: (str) Action, e.g. '11'

    Output:
        s_prime: new state produced by (s, a)
    '''

    s_prime = ''
    S = [int(x) for x in list(s)]
    col = int(a[0])
    n = int(a[1])
    S[col] = S[col] - n
    if S[col] < 0:
        return ''

    for i in S:
        s_prime += str(i)

    return s_prime


def reward(s, a, player):
    '''
    Calculate the reward of a move
    Input:
        s: (str) state
        a: (str) action
        player: (1 or -1) who makes move

    Output: (int) score
    '''

    if result(s, a) == '000':
        if player == -1:
            return 1000
        else:
            return -1000
    return 0


def update_Q(Q, state, action, alpha, gamma, player):
    '''
    Update the Q table
    Input:
        state: (str)
        action: (str)
        alpha: (float)
        gamma: (float)
        player: (1 or -1) who makes move

    Output:
        Q: updated Q
    '''

    # Player A's turn
    if player == 1:
        key = 'A' + state + action

        if Q.get(key) == None:
            Q[key] = 0
            return Q

        s_prime = result(state, action)
        actions = get_all_actions(s_prime)

        Q_prime = 10000000
        # If end state, Q' = 0
        if len(actions) == 0:
            Q_prime = 0

        # Else, find min of Q_prime
        for a in actions:
            key_prime = 'B' + s_prime + a

            if Q.get(key_prime) == None:
                Q[key_prime] = 0

            if Q[key_prime] < Q_prime:
                Q_prime = Q[key_prime]

        # Update Q
        r = reward(state, action, 1)
        Q[key] = Q[key] + alpha * (r + gamma*Q_prime - Q[key])

    # Player B's turn
    if player == -1:
        key = 'B' + state + action

        if Q.get(key) == None:
            Q[key] = 0
            return Q

        s_prime = result(state, action)
        actions = get_all_actions(s_prime)

        Q_prime = -10000000
        # If end state, Q' = 0
        if len(actions) == 0:
            Q_prime = 0

        # Else, find max of Q_prime
        for a in actions:
            key_prime = 'A' + s_prime + a

            if Q.get(key_prime) == None:
                Q[key_prime] = 0

            if Q[key_prime] > Q_prime:
                Q_prime = Q[key_prime]

        # Update Q
        r = reward(state, action, -1)
        Q[key] = Q[key] + alpha * (r + gamma*Q_prime - Q[key])

    return Q


def simulate_game(Q, state, alpha, gamma):
    '''
    Simulate one game
    Input:
        Q: Q table
        state: initial state
        alpha, gamma: parameter of Q function
    Output:
        Q: updated Q
    '''

    # Generate a random state-action pairs of a game
    game = []
    s = state
    while s != '000':
        a = random.choice(get_all_actions(s))
        game.append((s, a))
        s = result(s, a)

    # Run game
    player = 1
    for i, (s, a) in enumerate(game):
        if player == 1:
            Q = update_Q(Q, s, a, alpha, gamma, 1)
        if player == -1:
            Q = update_Q(Q, s, a, alpha, gamma, -1)

        # Switch turn
        player = - player

    return Q


def predicted_Q(s0, n=100000, alpha=1, gamma=0.9):
    '''
    Generate Q table with predicted score for each state-action pair
    Input:
        n: (int) how many games to simulate
        verbose: (bool) to debug

    Output:
        Q: Simulated Q table
    '''

    Q = {}
    for i in tqdm(range(n)):
        Q = simulate_game(Q, s0, alpha, gamma)
    return Q


def main():

    s0 = ''  # initial state
    Q = {}  # Q table
    play_again = 1  # to play as long as we want
    last_s0 = ''  # if users play last init state, don't have to generate Q again

    while play_again == 1:
        s0 = input('Game board: ')
        while len(s0) != 3:
            print('Game board must have format a-b-c (positive ints)')
            s0 = input('Game board: ')

        n = int(input('Number of games to simulate: '))

        if last_s0 != s0:
            Q = predicted_Q(s0, n)

        print('Initial board is {}-{}-{}, simulating {} games.'
              .format(s0[0], s0[1], s0[2], n))

        # Print Q table
        print('Final Q-values:')
        for key in sorted(Q):
            print('Q[{}, {}] = {:.3f}'.format(key[:-2], key[-2:], Q[key]))

        play_first = input('Do you want to play first (y/n): ')
        play_first = 1 if play_first == 'y' else 0

        player = 1
        s = s0

        while s != '000':
            best_move = ''
            actions = get_all_actions(s)

            # For Player A: π(s)=argmax{a}{Q[s,a]}
            if player == 1:
                # If play_first, A is human:
                if play_first:
                    print('Current Player: You')
                    print('Current Board: ', s)

                    best_move = input('Your next move: ')
                    while int(best_move[0]) >= 3 or int(best_move[0]) < 0:
                        print('Invalid move')
                        best_move = input('Your next move: ')

                    print('You chose ', best_move)

                # Else, A is bot. Choose best move
                else:
                    print('Current Player: Computer')
                    print('Current Board: ', s)

                    best_value = -10000000
                    for a in actions:
                        key = 'A' + s + a
                        if Q[key] > best_value:
                            best_value = Q[key]
                            best_move = a

                    print('Computer chose: ', best_move)

            # For Player B: π(s)=argmin{a}{Q[s,a]}
            else:
                # If not play_first, B is human
                if not play_first:
                    print('Current Player: You')
                    print('Current Board: ', s)

                    best_move = input('Your next move: ')
                    while int(best_move[0]) >= 3:
                        print('Invalid move')
                        best_move = input('Your next move: ')

                    print('You chose ', best_move)

                # Else, B is bot. Choose best move
                else:
                    print('Current Player: Computer')
                    print('Current Board: ', s)

                    best_value = 10000000
                    for a in actions:
                        key = 'B' + s + a
                        if Q[key] < best_value:
                            best_value = Q[key]
                            best_move = a
                    print('Computer chose: ', best_move)

            # Do action
            s_new = result(s, best_move)

            if s_new == '':
                print('Invalid move')
                continue

            # else
            s = s_new
            player = -player
            print()

        # End game
        print()
        print('Game Over.')

        # If player = 1 (or the play makes last move = -1)
        if player == 1:
            if play_first:
                print('You win.')
            else:
                print('Computer wins.')

        else:
            if not play_first:
                print('You win.')
            else:
                print('Computer wins.')

        play_again = int(input('Do you want to play another game (1/0): '))
        last_s0 = s0

    return 0


if __name__ == '__main__':
    main()
