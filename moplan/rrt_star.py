__all__ = ["BundlePlanner"]

__author__ 		= "Lekan Molu"
__copyright__ 	= "2022, RRT Star Planner from Pixels"
__credits__  	= "Alex Lamb, Shaoru Chen, Anurag Koul."
__license__ 	= "MSR Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__comments__    = "This code was written while under the brutal RSS 2022 deadline."
__loc__         = "Philadelphia, PA"
__date__ 		= "January 20, 2022"
__status__ 		= "Completed"


import nmslib
import torch
import random
import pickle
import argparse 
import numpy as np
from models import Encoder
from sklearn.cluster import KMeans
from .hj_prox import compute_hj_prox

parser = argparse.ArgumentParser(description='Latent State Planner')
parser.add_argument('--silent', '-si', action='store_true', help='silent debug print outs' )
parser.add_argument('--visualize', '-vz', action='store_false', help='visualize level sets?' )
parser.add_argument('--seed', '-sd', type=int, help='What seed to use?' )
parser.add_argument('--latent-dim', '-ld', type=int, default=256, help='size of the latent state?' )
parser.add_argument('--k_embedding_dim', '-ke', default=45, type=int, help='Embedding dimension' )
args = parser.parse_args()
args.verbose = True if not args.silent else False

torch.set_default_tensor_type(torch.FloatTensor)

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


class RRTStarPlannner():
    def __init__(self, state_init, state_goal, device):
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
        self.device     = device 

    def rrt_planner(self, model, enc, kmeanser): 
        """
            Author: Lekan Molu (Jan 20, 2022).
        """
        kmeans = kmeanser['kmeans']

        # load-dataset
        dataset = pickle.load(open(dataset_path, 'rb'))
        X, A, ast, est = dataset['X'], dataset['A'], dataset['ast'], dataset['est']

        # generate latent-states and find corresponding label
        latent_states, states_label = [], []
        for i in range(0, len(X), 256):
            with torch.no_grad():
                _latent_state  = enc(torch.FloatTensor(X[i:i + 256]).to(self.device))
                latent_states += _latent_state.cpu().numpy().tolist()
                states_label  += kmeans.predict(_latent_state.cpu().numpy().tolist()).tolist()

        "For each latent state, compute the proximal Hamiltonian"


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed(args.seed)

    # load model
    model = torch.load('../data/model.p', map_location=torch.device('cpu'))
    enc = Encoder(100 * 100, args.latent_dim).to(device)
    enc.load_state_dict(model['enc'])
    enc = enc.eval().to(device)

    # load clustering
    kmeanser = pickle.load(open('../data/kmeans_info.p', 'rb'))

    xi = torch.Tensor([0.0, 0.0])
    xg = torch.Tensor([1.0, 1.0])
    rrsp = RRTStarPlannner(xi, xg, device)

    rrsp.rrt_planner(model, enc, kmeanser)