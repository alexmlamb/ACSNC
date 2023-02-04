__all__ = ["BundlePlanner"]

__author__ 		= "Lekan Molu"
__copyright__ 	= "2022, Discrete Cosserat SoRO Analysis in Python"
__credits__  	= "Alex Lamb, Shaoru Chen, Anurag Koul."
__license__ 	= "MSR Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__comments__    = "This code was written while under the brutal RSS 2022 deadline."
__loc__         = "Philadelphia, PA"
__date__ 		= "January 20, 2022"
__status__ 		= "Completed"


import torch
import numpy as np

class Agent():
    def __init__(self, idx):
        if isinstance(idx, int):
            idx = f"".join(idx)
        self.agent_id = idx 

class Graph():
    def __init__(self, n, vertex_set, edges=None):
        """A graph (an undirected graph that is) that models
        the update equations of agents' positions on a state space
        (defined as a grid).

        The graph has a vertex set {1,2,...,n} so defined such that
        (i,j) is one of the graph's edges in case i and j are neighbors.
        This graph changes over time since the relationship between neighbors
        can change.

        Paramters
        =========
            n: number of initial birds (vertices) on this graph.
            .V: vertex_set, a set of vertices {1,2,...,n} that represent the labels
            of agents on a state space. Represent this as a list (see class vertex).
            .E: edges, a set of unordered pairs E = {(i,j): i,j \in V}.
                Edges have no self-loops i.e. i≠j or repeated edges (i.e. elements are distinct).
        """
        self.N = n
        if vertex_set is None:
            self.vertex_set = {f"{i+1}":Agent(i) for i in range(n)}
        else:
            self.vertex_set = {f"{i+1}":vertex_set[i] for i in range(n)}

        # edges are updated dynamically during game
        self.edges_set = edges

        # obtain the graph params
        self.reset(self.vertex_set[list(self.vertex_set.keys())[0]].w_e)

    def reset(self, w):
        # graph entities: this from Jadbabaie's paper
        self.Ap = np.zeros((self.N, self.N)) #adjacency matrix
        self.Dp = np.zeros((self.N, self.N)) #diagonal matrix of valencies
        self.θs = np.ones((self.N, 1))*w # agent headings
        self.I  = np.ones((self.N, self.N))
        self.Fp = np.zeros_like(self.Ap) # transition matrix for all the headings in this flock

    def insert_vertex(self, vertex):
        if isinstance(vertex, list):
            assert isinstance(vertex, Agent), "vertex to be inserted must be instance of class Vertex."
            for vertex_single in vertex:
                self.vertex_set[vertex_single.label] = vertex_single.neighbors
        else:
            self.vertex_set[vertex.label] = vertex

    def insert_edge(self, from_vertices, to_vertices):
        if isinstance(from_vertices, list) and isinstance(to_vertices, list):
            for from_vertex, to_vertex in zip(from_vertices, to_vertices):
                self.insert_edge(from_vertex, to_vertex)
            return
        else:
            assert isinstance(from_vertices, Agent), "from_vertex to be inserted must be instance of class Vertex."
            assert isinstance(to_vertices, Agent), "to_vertex to be inserted must be instance of class Vertex."
            from_vertices.update_neighbor(to_vertices)
            self.vertex_set[from_vertices.label] = from_vertices.neighbors
            self.vertex_set[to_vertices.label] = to_vertices.neighbors

    def adjacency_matrix(self, t):
        for i in range(self.Ap.shape[0]):
            for j in range(self.Ap.shape[1]):
                for verts in sorted(self.vertex_set.keys()):
                    if str(j) in self.vertex_set[verts].neighbors:
                        self.Ap[i,j] = 1
        return self.Ap

    def diag_matrix(self):
        "build Dp matrix"
        i=0
        for vertex, egdes in self.vertex_set.items():
            self.Dp[i,i] = self.vertex_set[vertex].valence
        return self.Dp

    def update_headings(self, t):
        return self.adjacency_matrix(t)@self.θs
        
class PriorityQueue():
    def __init__(self, queue_len=None):
        if queue_len is not None:
            self.queue = [x for x in range(queue_len)]
        else:
            self.queue = []

    def push(self, fiber, push_idx=None):
        if push_idx is not None:
            self.queue[push_idx] = fiber
        else:
            self.queue.append(fiber)

    def pop(self):
        popped = self.queue[-1]

        self.queue = self.queue[:-1]

        return popped


class BundlePlannner():
    def __init__(self, state_init, state_goal):
        """
            A fiber bundle planner in latent spaces. Adapted from the following papers:

            (i) A Orthey, S Akbar and M Toussaint, Multilevel Motion Planning: A Fiber Bundle Formulation, 2020. Preprint. Also available at https://arxiv.org/abs/2007.09435.
            (ii) A Orthey and M Toussaint, Rapidly-Exploring Quotient-Space Trees: Motion Planning using Sequential Simplifications, ISRR, 2019. Also available at https://arxiv.org/abs/1906.01350.
            (iii) A Orthey, A Escande and E Yoshida, Quotient Space Motion Planning, ICRA, 2018. Also available at https://arxiv.org/abs/1807.09468.

            Input:
                .state_init: $x_0$ 
                            A 2-D (floating point) tensor containing the position of the starting configuration in pixel space.
                .state_goal: $x_g$ 
                            A 2-D (floating point) tensor containing the position of the goal configuration in pixel space.
        """
        self.queue = PriorityQueue()

        self.state_init = state_init
        self.state_goal = state_goal

    # def planner(self, state_init, state_goal, img_encoder, latent_fwd, latent2state, state2img):  # Alex's original
    def bundle_planner(self, img_encoder, latent_fwd, latent2state, clusters): 
        """
            Inputs
            ======
            .img_encoder: Enc: O --> z
                        A mapping operator from the obs_space of the agent to latent states i.e. 1-D tensor (1,256) dim.
            .latent_fwd: \dot{z}(t) = f(t; z, u)
                        Forward dynamics model in the latent space.
            .latent2state: x(t) = g(t; z)
                        Given a latent state, predict the (intrinsic) state (containing the pixel coordinates).
            [.state2img: S2I: x --> O
                        Given the 2D coordinates of a state, generate an image that is [100, 100].
                        (why do we need this inverse RL formulation again?]
            .clusters: A list of discretized clusters in which the latent spaces exist.

            Author: Lekan Molu (Jan 20, 2022).
        """

        "First initialize a priority queue sorted by the importance of each bundle space."
        for clu_idx in range(len(clusters)):
            self.find_section(clusters[clu_idx])
            self.queue.push(clusters[clu_idx])

            while not self.reach_cluster_goal(clusters[clu_idx]):
                "Select the most important bundle space and grow the graph or tree"
                clu_sel =   self.queue.pop()
                self.grow_graph(clu_sel)
                self.queue.push(clu_sel)
        
        # action = torch.zeros((1,2))
        # return action
    
    def reach_cluster_goal(self, cur_fiber, goal_fiber):
        """
            Check if we have reached the goal bundle 'goal_fiber' 
            when we start in a current cluster 'cur_fiber'.
        """


if __name__ == "__main__":

    s0 = torch.Tensor([[0.4, 0.4]])
    sg = torch.Tensor([[0.6,0.6]])

    img_encoder = lambda img: torch.zeros((1,256))
    state2img = lambda state: torch.zeros((1,100*100))
    latent_fwd = lambda state,action: state*0.0
    latent2state = lambda latent: torch.zeros((1,2))

    plan(s0, sg, img_encoder, state2img, latent_fwd, latent2state)

