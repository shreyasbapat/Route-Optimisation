import numpy as np
import random
from vprtw_aco_figure import VrptwAcoFigure
from vrptw_base import VrptwGraph, PathMessage
from ant import Ant
from threading import Thread, Event
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import copy
import time
from multiprocessing import Process
from multiprocessing import Queue as MPQueue


class MultipleAntColonySystem:
    def __init__(
        self,
        graph: VrptwGraph,
        ants_num=10,
        beta=1,
        q0=0.1,
        whether_or_not_to_show_figure=True,
    ):
        super()
        self.graph = graph
        self.ants_num = ants_num
        self.max_load = graph.vehicle_capacity
        self.beta = beta
        self.q0 = q0
        # best path
        self.best_path_distance = None
        self.best_path = None
        self.best_vehicle_num = None

        self.whether_or_not_to_show_figure = whether_or_not_to_show_figure

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

    @staticmethod
    def new_active_ant(
        ant: Ant,
        vehicle_num: int,
        local_search: bool,
        IN: np.numarray,
        q0: float,
        beta: int,
        stop_event: Event,
    ):

        # print('[new_active_ant]: start, start_index %d' % ant.travel_path[0])

        unused_depot_count = vehicle_num

        while not ant.index_to_visit_empty() and unused_depot_count > 0:
            if stop_event.is_set():
                # print('[new_active_ant]: receive stop event')
                return

            next_index_meet_constrains = ant.cal_next_index_meet_constrains()

            if len(next_index_meet_constrains) == 0:
                ant.move_to_next_index(0)
                unused_depot_count -= 1
                continue

            length = len(next_index_meet_constrains)
            ready_time = np.zeros(length)
            due_time = np.zeros(length)

            for i in range(length):
                ready_time[i] = ant.graph.nodes[
                    next_index_meet_constrains[i]
                ].ready_time
                due_time[i] = ant.graph.nodes[next_index_meet_constrains[i]].due_time

            delivery_time = np.maximum(
                ant.vehicle_travel_time
                + ant.graph.node_dist_mat[ant.current_index][
                    next_index_meet_constrains
                ],
                ready_time,
            )
            delta_time = delivery_time - ant.vehicle_travel_time
            distance = delta_time * (due_time - ant.vehicle_travel_time)

            distance = np.maximum(1.0, distance - IN[next_index_meet_constrains])
            closeness = 1 / distance

            transition_prob = ant.graph.pheromone_mat[ant.current_index][
                next_index_meet_constrains
            ] * np.power(closeness, beta)
            transition_prob = transition_prob / np.sum(transition_prob)

            if np.random.rand() < q0:
                max_prob_index = np.argmax(transition_prob)
                next_index = next_index_meet_constrains[max_prob_index]
            else:

                next_index = MultipleAntColonySystem.stochastic_accept(
                    next_index_meet_constrains, transition_prob
                )

            ant.graph.local_update_pheromone(ant.current_index, next_index)
            ant.move_to_next_index(next_index)
        if ant.index_to_visit_empty():
            ant.graph.local_update_pheromone(ant.current_index, 0)
            ant.move_to_next_index(0)

        ant.insertion_procedure(stop_event)

        if local_search is True and ant.index_to_visit_empty():
            ant.local_search_procedure(stop_event)

    @staticmethod
    def acs_time(
        new_graph: VrptwGraph,
        vehicle_num: int,
        ants_num: int,
        q0: float,
        beta: int,
        global_path_queue: Queue,
        path_found_queue: Queue,
        stop_event: Event,
    ):

        print("[acs_time]: start, vehicle_num %d" % vehicle_num)
        global_best_path = None
        global_best_distance = None
        ants_pool = ThreadPoolExecutor(ants_num)
        ants_thread = []
        ants = []
        while True:
            print("[acs_time]: new iteration")

            if stop_event.is_set():
                print("[acs_time]: receive stop event")
                return

            for k in range(ants_num):
                ant = Ant(new_graph, 0)
                thread = ants_pool.submit(
                    MultipleAntColonySystem.new_active_ant,
                    ant,
                    vehicle_num,
                    True,
                    np.zeros(new_graph.node_num),
                    q0,
                    beta,
                    stop_event,
                )
                ants_thread.append(thread)
                ants.append(ant)

            for thread in ants_thread:
                thread.result()

            ant_best_travel_distance = None
            ant_best_path = None
            for ant in ants:

                if stop_event.is_set():
                    print("[acs_time]: receive stop event")
                    return

                if not global_path_queue.empty():
                    info = global_path_queue.get()
                    while not global_path_queue.empty():
                        info = global_path_queue.get()
                    print("[acs_time]: receive global path info")
                    global_best_path, global_best_distance, global_used_vehicle_num = (
                        info.get_path_info()
                    )

                if ant.index_to_visit_empty() and (
                    ant_best_travel_distance is None
                    or ant.total_travel_distance < ant_best_travel_distance
                ):
                    ant_best_travel_distance = ant.total_travel_distance
                    ant_best_path = ant.travel_path

            new_graph.global_update_pheromone(global_best_path, global_best_distance)

            if (
                ant_best_travel_distance is not None
                and ant_best_travel_distance < global_best_distance
            ):
                print(
                    "[acs_time]: ants' local search found a improved feasible path, send path info to macs"
                )
                path_found_queue.put(
                    PathMessage(ant_best_path, ant_best_travel_distance)
                )

            ants_thread.clear()
            for ant in ants:
                ant.clear()
                del ant
            ants.clear()

    @staticmethod
    def acs_vehicle(
        new_graph: VrptwGraph,
        vehicle_num: int,
        ants_num: int,
        q0: float,
        beta: int,
        global_path_queue: Queue,
        path_found_queue: Queue,
        stop_event: Event,
    ):
        print("[acs_vehicle]: start, vehicle_num %d" % vehicle_num)
        global_best_path = None
        global_best_distance = None

        current_path, current_path_distance, _ = new_graph.nearest_neighbor_heuristic(
            max_vehicle_num=vehicle_num
        )

        current_index_to_visit = list(range(new_graph.node_num))
        for ind in set(current_path):
            current_index_to_visit.remove(ind)

        ants_pool = ThreadPoolExecutor(ants_num)
        ants_thread = []
        ants = []
        IN = np.zeros(new_graph.node_num)
        while True:
            print("[acs_vehicle]: new iteration")

            if stop_event.is_set():
                print("[acs_vehicle]: receive stop event")
                return

            for k in range(ants_num):
                ant = Ant(new_graph, 0)
                thread = ants_pool.submit(
                    MultipleAntColonySystem.new_active_ant,
                    ant,
                    vehicle_num,
                    False,
                    IN,
                    q0,
                    beta,
                    stop_event,
                )

                ants_thread.append(thread)
                ants.append(ant)

            for thread in ants_thread:
                thread.result()

            for ant in ants:

                if stop_event.is_set():
                    print("[acs_vehicle]: receive stop event")
                    return

                IN[ant.index_to_visit] = IN[ant.index_to_visit] + 1

                if len(ant.index_to_visit) < len(current_index_to_visit):
                    current_path = copy.deepcopy(ant.travel_path)
                    current_index_to_visit = copy.deepcopy(ant.index_to_visit)
                    current_path_distance = ant.total_travel_distance

                    IN = np.zeros(new_graph.node_num)

                    if ant.index_to_visit_empty():
                        print(
                            "[acs_vehicle]: found a feasible path, send path info to macs"
                        )
                        path_found_queue.put(
                            PathMessage(ant.travel_path, ant.total_travel_distance)
                        )

            new_graph.global_update_pheromone(current_path, current_path_distance)

            if not global_path_queue.empty():
                info = global_path_queue.get()
                while not global_path_queue.empty():
                    info = global_path_queue.get()
                print("[acs_vehicle]: receive global path info")
                global_best_path, global_best_distance, global_used_vehicle_num = (
                    info.get_path_info()
                )

            new_graph.global_update_pheromone(global_best_path, global_best_distance)

            ants_thread.clear()
            for ant in ants:
                ant.clear()
                del ant
            ants.clear()

    def run_multiple_ant_colony_system(self, file_to_write_path=None):
        path_queue_for_figure = MPQueue()
        multiple_ant_colony_system_thread = Process(
            target=self._multiple_ant_colony_system,
            args=(path_queue_for_figure, file_to_write_path),
        )
        multiple_ant_colony_system_thread.start()

        if self.whether_or_not_to_show_figure:
            figure = VrptwAcoFigure(self.graph.nodes, path_queue_for_figure)
            figure.run()
        multiple_ant_colony_system_thread.join()

    def _multiple_ant_colony_system(
        self, path_queue_for_figure: MPQueue, file_to_write_path=None
    ):
        if file_to_write_path is not None:
            file_to_write = open(file_to_write_path, "w")
        else:
            file_to_write = None

        start_time_total = time.time()

        global_path_to_acs_time = Queue()
        global_path_to_acs_vehicle = Queue()

        path_found_queue = Queue()

        self.best_path, self.best_path_distance, self.best_vehicle_num = (
            self.graph.nearest_neighbor_heuristic()
        )
        path_queue_for_figure.put(PathMessage(self.best_path, self.best_path_distance))

        while True:
            print("[multiple_ant_colony_system]: new iteration")
            start_time_found_improved_solution = time.time()

            global_path_to_acs_vehicle.put(
                PathMessage(self.best_path, self.best_path_distance)
            )
            global_path_to_acs_time.put(
                PathMessage(self.best_path, self.best_path_distance)
            )

            stop_event = Event()

            graph_for_acs_vehicle = self.graph.copy(self.graph.init_pheromone_val)
            acs_vehicle_thread = Thread(
                target=MultipleAntColonySystem.acs_vehicle,
                args=(
                    graph_for_acs_vehicle,
                    self.best_vehicle_num - 1,
                    self.ants_num,
                    self.q0,
                    self.beta,
                    global_path_to_acs_vehicle,
                    path_found_queue,
                    stop_event,
                ),
            )

            graph_for_acs_time = self.graph.copy(self.graph.init_pheromone_val)
            acs_time_thread = Thread(
                target=MultipleAntColonySystem.acs_time,
                args=(
                    graph_for_acs_time,
                    self.best_vehicle_num,
                    self.ants_num,
                    self.q0,
                    self.beta,
                    global_path_to_acs_time,
                    path_found_queue,
                    stop_event,
                ),
            )

            print("[macs]: start acs_vehicle and acs_time")
            acs_vehicle_thread.start()
            acs_time_thread.start()

            best_vehicle_num = self.best_vehicle_num

            while acs_vehicle_thread.is_alive() and acs_time_thread.is_alive():

                given_time = 10
                if time.time() - start_time_found_improved_solution > 60 * given_time:
                    stop_event.set()
                    self.print_and_write_in_file(file_to_write, "*" * 50)
                    self.print_and_write_in_file(
                        file_to_write,
                        "time is up: cannot find a better solution in given time(%d minutes)"
                        % given_time,
                    )
                    self.print_and_write_in_file(
                        file_to_write,
                        "it takes %0.3f second from multiple_ant_colony_system running"
                        % (time.time() - start_time_total),
                    )
                    self.print_and_write_in_file(
                        file_to_write, "the best path have found is:"
                    )
                    self.print_and_write_in_file(file_to_write, self.best_path)
                    self.print_and_write_in_file(
                        file_to_write,
                        "best path distance is %f, best vehicle_num is %d"
                        % (self.best_path_distance, self.best_vehicle_num),
                    )
                    self.print_and_write_in_file(file_to_write, "*" * 50)

                    if self.whether_or_not_to_show_figure:
                        path_queue_for_figure.put(PathMessage(None, None))

                    if file_to_write is not None:
                        file_to_write.flush()
                        file_to_write.close()
                    return

                if path_found_queue.empty():
                    continue

                path_info = path_found_queue.get()
                print("[macs]: receive found path info")
                found_path, found_path_distance, found_path_used_vehicle_num = (
                    path_info.get_path_info()
                )
                while not path_found_queue.empty():
                    path, distance, vehicle_num = path_found_queue.get().get_path_info()

                    if distance < found_path_distance:
                        found_path, found_path_distance, found_path_used_vehicle_num = (
                            path,
                            distance,
                            vehicle_num,
                        )

                    if vehicle_num < found_path_used_vehicle_num:
                        found_path, found_path_distance, found_path_used_vehicle_num = (
                            path,
                            distance,
                            vehicle_num,
                        )

                if found_path_distance < self.best_path_distance:

                    start_time_found_improved_solution = time.time()

                    self.print_and_write_in_file(file_to_write, "*" * 50)
                    self.print_and_write_in_file(
                        file_to_write,
                        "[macs]: distance of found path (%f) better than best path's (%f)"
                        % (found_path_distance, self.best_path_distance),
                    )
                    self.print_and_write_in_file(
                        file_to_write,
                        "it takes %0.3f second from multiple_ant_colony_system running"
                        % (time.time() - start_time_total),
                    )
                    self.print_and_write_in_file(file_to_write, "*" * 50)
                    if file_to_write is not None:
                        file_to_write.flush()

                    self.best_path = found_path
                    self.best_vehicle_num = found_path_used_vehicle_num
                    self.best_path_distance = found_path_distance

                    if self.whether_or_not_to_show_figure:
                        path_queue_for_figure.put(
                            PathMessage(self.best_path, self.best_path_distance)
                        )

                    global_path_to_acs_vehicle.put(
                        PathMessage(self.best_path, self.best_path_distance)
                    )
                    global_path_to_acs_time.put(
                        PathMessage(self.best_path, self.best_path_distance)
                    )

                if found_path_used_vehicle_num < best_vehicle_num:

                    start_time_found_improved_solution = time.time()
                    self.print_and_write_in_file(file_to_write, "*" * 50)
                    self.print_and_write_in_file(
                        file_to_write,
                        "[macs]: vehicle num of found path (%d) better than best path's (%d), found path distance is %f"
                        % (
                            found_path_used_vehicle_num,
                            best_vehicle_num,
                            found_path_distance,
                        ),
                    )
                    self.print_and_write_in_file(
                        file_to_write,
                        "it takes %0.3f second multiple_ant_colony_system running"
                        % (time.time() - start_time_total),
                    )
                    self.print_and_write_in_file(file_to_write, "*" * 50)
                    if file_to_write is not None:
                        file_to_write.flush()

                    self.best_path = found_path
                    self.best_vehicle_num = found_path_used_vehicle_num
                    self.best_path_distance = found_path_distance

                    if self.whether_or_not_to_show_figure:
                        path_queue_for_figure.put(
                            PathMessage(self.best_path, self.best_path_distance)
                        )

                    print("[macs]: send stop info to acs_time and acs_vehicle")
                    stop_event.set()

    @staticmethod
    def print_and_write_in_file(file_to_write=None, message="default message"):
        if file_to_write is None:
            print(message)
        else:
            print(message)
            file_to_write.write(str(message) + "\n")
