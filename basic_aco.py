import numpy as np
import random
from vprtw_aco_figure import VrptwAcoFigure
from vrptw_base import VrptwGraph, PathMessage
from ant import Ant
from threading import Thread
from queue import Queue
import time


class BasicACO:
    def __init__(
        self,
        graph: VrptwGraph,
        ants_num=10,
        max_iter=200,
        beta=2,
        q0=0.1,
        whether_or_not_to_show_figure=True,
    ):
        super()
        self.graph = graph
        self.ants_num = ants_num
        self.max_iter = max_iter
        self.max_load = graph.vehicle_capacity
        self.beta = beta
        self.q0 = q0
        # best path
        self.best_path_distance = None
        self.best_path = None
        self.best_vehicle_num = None

        self.whether_or_not_to_show_figure = whether_or_not_to_show_figure

    def run_basic_aco(self):
        path_queue_for_figure = Queue()
        basic_aco_thread = Thread(target=self._basic_aco, args=(path_queue_for_figure,))
        basic_aco_thread.start()

        if self.whether_or_not_to_show_figure:
            figure = VrptwAcoFigure(self.graph.nodes, path_queue_for_figure)
            figure.run()
        basic_aco_thread.join()

        if self.whether_or_not_to_show_figure:
            path_queue_for_figure.put(PathMessage(None, None))

    def _basic_aco(self, path_queue_for_figure: Queue):
        start_time_total = time.time()

        start_iteration = 0
        for iter in range(self.max_iter):

            ants = list(Ant(self.graph) for _ in range(self.ants_num))
            for k in range(self.ants_num):
                while not ants[k].index_to_visit_empty():
                    next_index = self.select_next_index(ants[k])
                    if not ants[k].check_condition(next_index):
                        next_index = self.select_next_index(ants[k])
                        if not ants[k].check_condition(next_index):
                            next_index = 0

                    ants[k].move_to_next_index(next_index)
                    self.graph.local_update_pheromone(ants[k].current_index, next_index)

                ants[k].move_to_next_index(0)
                self.graph.local_update_pheromone(ants[k].current_index, 0)

            paths_distance = np.array([ant.total_travel_distance for ant in ants])

            best_index = np.argmin(paths_distance)
            if (
                self.best_path is None
                or paths_distance[best_index] < self.best_path_distance
            ):
                self.best_path = ants[int(best_index)].travel_path
                self.best_path_distance = paths_distance[best_index]
                self.best_vehicle_num = self.best_path.count(0) - 1
                start_iteration = iter

                if self.whether_or_not_to_show_figure:
                    path_queue_for_figure.put(
                        PathMessage(self.best_path, self.best_path_distance)
                    )

                print("\n")
                print(
                    "[iteration %d]: find a improved path, its distance is %f"
                    % (iter, self.best_path_distance)
                )
                print(
                    "it takes %0.3f second multiple_ant_colony_system running"
                    % (time.time() - start_time_total)
                )

            self.graph.global_update_pheromone(self.best_path, self.best_path_distance)

            given_iteration = 100
            if iter - start_iteration > given_iteration:
                print("\n")
                print(
                    "iteration exit: can not find better solution in %d iteration"
                    % given_iteration
                )
                break

        print("\n")
        print(
            "final best path distance is %f, number of vehicle is %d"
            % (self.best_path_distance, self.best_vehicle_num)
        )
        print(
            "it takes %0.3f second multiple_ant_colony_system running"
            % (time.time() - start_time_total)
        )

    def select_next_index(self, ant):
        current_index = ant.current_index
        index_to_visit = ant.index_to_visit

        transition_prob = self.graph.pheromone_mat[current_index][
            index_to_visit
        ] * np.power(
            self.graph.heuristic_info_mat[current_index][index_to_visit], self.beta
        )
        transition_prob = transition_prob / np.sum(transition_prob)

        if np.random.rand() < self.q0:
            max_prob_index = np.argmax(transition_prob)
            next_index = index_to_visit[max_prob_index]
        else:
            next_index = BasicACO.stochastic_accept(index_to_visit, transition_prob)
        return next_index

    @staticmethod
    def stochastic_accept(index_to_visit, transition_prob):
        # calculate N and max fitness value
        N = len(index_to_visit)

        # normalize
        sum_tran_prob = np.sum(transition_prob)
        norm_transition_prob = transition_prob / sum_tran_prob

        # select: O(1)
        while True:
            # randomly select an individual with uniform probability
            ind = int(N * random.random())
            if random.random() <= norm_transition_prob[ind]:
                return index_to_visit[ind]
