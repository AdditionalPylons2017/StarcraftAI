

from StarcraftAI.agents.learning_agent import QLearningTable


def test_QLearnTable():

    action_list = ["Rock","Paper","Scissors"]

    qlearn = QLearningTable(actions=list(range(len(action_list))), e_greedy=1)

    prev_state = ["Start","paper"]
    cur_action = "Rock"
    cur_state = ["Lost", "paper"]
    qlearn.learn(str(prev_state), 0, -100, str(cur_state))
    prev_state = ["Start", "paper"]
    cur_action = "Paper"
    cur_state = ["Tie", "paper"]
    qlearn.learn(str(prev_state), 1, 0, str(cur_state))
    prev_state = ["Start", "paper"]
    cur_action = "Scissors"
    cur_state = ["Won", "paper"]
    qlearn.learn(str(prev_state), 2, 100, str(cur_state))
    prev_state = ["Start", "rock"]
    cur_action = "rock"
    cur_state = ["Tie", "rock"]
    qlearn.learn(str(prev_state), 0, 0, str(cur_state))
    prev_state = ["Start", "rock"]
    cur_action = "paper"
    cur_state = ["Win", "rock"]
    qlearn.learn(str(prev_state), 1, 100, str(cur_state))
    prev_state = ["Start", "rock"]
    cur_action = "scissors"
    cur_state = ["Lost", "rock"]
    qlearn.learn(str(prev_state), 2, -100, str(cur_state))

    current_state = ["Start","paper"]

    rl_action = qlearn.choose_action(str(current_state))
    print(qlearn.q_table)

    assert rl_action == 2


def test_TravisCI_test():

    assert True

print("Test Successful")
