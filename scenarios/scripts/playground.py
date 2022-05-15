import pickle
import matplotlib.pyplot as plt
import igp2 as ip

ego = pickle.load(open("3", "rb"))
cost_factors = {"time": 1.0, "velocity": 0.0, "acceleration": 1.0, "jerk": 0., "heading": 0,
                "angular_velocity": 0.0, "angular_acceleration": 0., "curvature": 0.0, "safety": 0.}
cost = ip.Cost(cost_factors)
scenario_map = ip.Map.parse_from_opendrive("scenarios/maps/scenario2.xodr")
preds = ego.bayesian_network.tree.predictions
for aid, pred in preds.items():
    pred.plot(scenario_map, 1, cost)
    plt.show()
for rr in ego.bayesian_network.results.mcts_results:
    continue
for action in ego.bayesian_network.variables["omega_1"]:
    ego.explain_action({"omega_1": action})
