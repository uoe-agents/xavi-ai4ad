import sys
from typing import Dict, List, Tuple

import logging
from shapely.geometry import Polygon

import igp2 as ip
import xavi
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

from scenarios.scripts.util import generate_random_frame, setup_xavi_logging
from xavi import XAVIInference


if __name__ == '__main__':
    setup_xavi_logging()

    # Set run parameters here
    seed = 42
    max_speed = 12.0
    ego_id = 0
    n_simulations = 15
    fps = 20  # Simulator frequency
    T = 2  # MCTS update period

    random.seed(seed)
    np.random.seed(42)
    np.seterr(divide="ignore")
    ip.Maneuver.MAX_SPEED = max_speed  # TODO: add global method to igp2 to set all possible parameters

    # Set randomised spawn parameters here
    ego_spawn_box = ip.Box(np.array([-65.0, -1.8]), 10, 3.5, 0.0)
    ego_vel_range = (5.0, max_speed)
    veh1_spawn_box = ip.Box(np.array([-60.0, 1.7]), 10, 3.5, 0.0)
    veh1_vel_range = (5.0, max_speed)
    veh2_spawn_box = ip.Box(np.array([-18.34, -25.5]), 3.5, 10, 0.0)
    veh2_vel_range = (5.0, max_speed)

    # Vehicle goals
    goals = {
        ego_id: ip.BoxGoal(ip.Box(np.array([-6.0, 0.0]), 5, 7, 0.0)),
        ego_id + 1: ip.BoxGoal(ip.Box(np.array([-22, -25.5]), 3.5, 5, 0.0)),
        ego_id + 2: ip.BoxGoal(ip.Box(np.array([-6.0, 0.0]), 5, 7, 0.0))
    }

    scenario_path = "scenarios/maps/scenario1.xodr"
    scenario_map = ip.Map.parse_from_opendrive(scenario_path)

    frame = generate_random_frame(ego_id,
                                  scenario_map,
                                  [(ego_spawn_box, ego_vel_range),
                                   (veh1_spawn_box, veh1_vel_range),
                                   (veh2_spawn_box, veh2_vel_range)])

    # ip.plot_map(scenario_map, markings=True, midline=True)
    # plt.plot(*list(zip(*ego_spawn_box.boundary)))
    # plt.plot(*list(zip(*veh1_spawn_box.boundary)))
    # plt.plot(*list(zip(*veh2_spawn_box.boundary)))
    # for aid, state in frame.items():
    #     plt.plot(*state.position, marker="x")
    #     plt.text(*state.position, aid)
    # for goal in goals.values():
    #     plt.plot(*list(zip(*goal.box.boundary)), c="g")
    # plt.gca().add_patch(plt.Circle(frame[0].position, 100, color='b', fill=False))
    # plt.show()

    cost_factors = {"time": 0.1, "velocity": 0.0, "acceleration": 0.0, "jerk": 0., "heading": 10,
                    "angular_velocity": 0.0, "angular_acceleration": 0., "curvature": 0.0, "safety": 0.}

    ego = xavi.XAVIAgent(agent_id=ego_id,
                         initial_state=frame[ego_id],
                         t_update=T,
                         scenario_map=scenario_map,
                         goal=goals[ego_id],
                         cost_factors=cost_factors,
                         fps=fps,
                         n_simulations=n_simulations,
                         view_radius=100,
                         store_results="final")
    obs = ip.Observation(frame, scenario_map)
    ego.update_observations(obs)
    ego.get_goals(obs)
    ego.update_plan(obs)

    bayesian_network = ego.bayesian_network
    sample = bayesian_network.tree.possible_samples[0]
    factions = {"omega_1": 'ChangeLaneLeft()',
                # "omega_2": 'Exit(turn_target: ([-14.09403104,   1.74012177]))',
                # "omega_3": 'Continue(termination_point: ([-6.00122365,  1.7457941 ]))'
                }
    cfactions = {"omega_1": 'Exit(turn_target: ([-14.09157785,  -1.75987737]))',
                 # "omega_2": 'Continue(termination_point: ([-5.99877046, -1.75420504]))',
                 # "omega_3": 'Continue(termination_point: ([-6.00122365,  1.7457941 ]))'
                 }
    rewards = {"time": ip.Reward().time_discount ** 7.5,
               "coll": None}
    outcome = "done"

    explanation = ego.explain_action(cfactions)

    bn = bayesian_network.to_bayesian_network()
    inf = XAVIInference(bn)
    var = inf.rank_agent_influence()
    diffs = inf.mean_differences(variables=[f"reward_{comp}" for comp in rewards],
                                 factual=factions,
                                 counterfactual=cfactions)
    p_outcome, outcome = inf.most_likely_outcome(factions)

    g = ego.mcts.results.tree.graph
    pos = xavi.hierarchy_pos(g, root=("Root",))
    nx.draw(g, pos, with_labels=False)
    nx.draw_networkx_edge_labels(g, pos, font_color='red', rotate=False)
    plt.show()

    ego.bayesian_network.p_r_omega(['Root', 'ChangeLaneLeft()', 'Exit(turn_target: ([-14.09403104,   1.74012177]))',
                                    'Continue(termination_point: ([-6.00122365,  1.7457941 ]))'], True, time=-8)

    # carla_sim = ip.carla.CarlaSim(xodr=scenario_path,
    #                               carla_path="C:\\Carla")
    #
    # agents = {}
    # agents_meta = ip.AgentMetadata.default_meta_frame(frame)
    # for aid in frame.keys():
    #     goal = goals[aid]
    #
    #     if aid == ego_id:
    #         agents[aid] = xavi.XAVIAgent(agent_id=aid,
    #                                      initial_state=frame[aid],
    #                                      t_update=T,
    #                                      scenario_map=scenario_map,
    #                                      goal=goal,
    #                                      cost_factors=cost_factors,
    #                                      fps=fps,
    #                                      n_simulations=n_simulations,
    #                                      view_radius=100,
    #                                      store_results="all")
    #         rolename = "ego"
    #     else:
    #         agents[aid] = ip.carla.TrafficAgent(aid, frame[aid], goal, fps)
    #         agents[aid].set_destination(goal, scenario_map)
    #         rolename = None
    #
    #     carla_sim.add_agent(agents[aid], rolename)
    #
    # visualiser = ip.carla.Visualiser(carla_sim)
    # visualiser.run()
