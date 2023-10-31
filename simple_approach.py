# For each player, calculate number of wins and losses
from sklearn import tree
player_stats = {}

f = open("training_data.txt")
for data in f:
    split = data.split(",")
    first_team_players = split[:5]
    second_team_players = split[5:10]
    score = int(split[-1].strip())

    # for each player in the first team update the win and loss counts
    for player in first_team_players:
        if player in player_stats.keys():
            player_stats[player][0] += 2-score
            player_stats[player][1] += score-1
        else:
            player_stats[player] = [2-score, score-1]
    # for each player in the second team update the win and loss counts
    for player in second_team_players:
        if player in player_stats.keys():
            player_stats[player][0] += score-1
            player_stats[player][1] += 2-score
        else:
            player_stats[player] = [score-1, 2-score]
f.close()

# Include win ratio to player_stats dict
for player in player_stats.keys():
    win_ratio = player_stats[player][0] / \
        (player_stats[player][0]+player_stats[player][1])
    player_stats[player].append(win_ratio)


# read train data and convert each player to #wins, #loss, win_percentage as features
f = open("training_data.txt")
training_data = []
labels = []

for data in f:
    split = data.split(",")
    players = split[:10]
    score = int(split[-1].strip())

    training_example = []
    for player in players:
        training_example.append(
            player_stats[player][0]+player_stats[player][1])
        training_example.append(player_stats[player][2])

    training_data.append(training_example)
    labels.append(score)

f.close()

# Fit a simple Decision Tree Model


dtc = tree.DecisionTreeClassifier()
dtc.fit(training_data, labels)

# Prepare test data
f = open("test_data.txt")

test_data = []

for data in f:
    players = data.strip().split(",")
    # Transform test data into #wins, #losses, win_percentage format
    example = []
    for player in players:
        example.append(player_stats[player][0]+player_stats[player][1])
        example.append(player_stats[player][2])

    test_data.append(example)

f.close()

# Calculate predictions
predictions = dtc.predict(test_data)


def accuracy(true, pred):
    correct_ct = 0
    total_ct = len(true)

    for t, p in zip(true, pred):
        if t == p:
            correct_ct += 1

    return correct_ct / total_ct


expected_output = []
f = open("expected_output.txt")
for line in f:
    expected_output.append(int(line.strip()))

acc = accuracy(expected_output, predictions)
print(f"Accuracy is {acc:.2f}")
