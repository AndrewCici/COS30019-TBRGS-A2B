from typing import Dict, Tuple, Optional, List

# Create the Node object
class Node:
    def __init__(self, node_id: int, x: float, y: float):
        self.id = node_id
        self.x = x
        self.y = y
# String representation of the node object eg: Node(1)
    def __repr__(self):
        return f"Node({self.id})"

# Graph respresents route-finding problem
class Graph:
    def __init__(self):
        self.nodes: Dict[int, Node] = {}  # Dict maps node id to node object
        self.adj: Dict[int, List[Tuple[int, float]]] = {}  # adj is a dictionary with key is nodeid and values is list of tuples with neighbour node and cost to that
        self.origin: Optional[int] = None # Stores the ID of the origin node
        self.destinations: List[int] = [] # Stores the list of destinations

# Method to add node to graph
    def add_node(self, node: Node):
        self.nodes[node.id] = node # Put nodes in dictionary keyed by id
        self.adj[node.id] = []   # Initialize adjacency list

# Method to add edge
    def add_edge(self, from_id: int, to_id: int, cost: float):
        self.adj[from_id].append((to_id, cost)) #append tuple to adjacency list of source node

# File parser
def load_graph(filename: str):
    graph = Graph()
    section = None
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Section headers
            if line == "Nodes:":
                section = "nodes"
                continue
            elif line == "Edges:":
                section = "edges"
                continue
            elif line == "Origin:":
                section = "origin"
                continue
            elif line == "Destinations:":
                section = "destinations"
                continue

            # Process based on current section
            if section == "nodes":
                # Format: "1: (4,1)"
                node_id_str, coords_str = line.split(':')
                node_id = int(node_id_str.strip())
                coords_str = coords_str.strip().strip('()')
                x_str, y_str = coords_str.split(',')
                x, y = float(x_str), float(y_str)
                graph.add_node(Node(node_id, x, y))

            elif section == "edges":
                # Format: "(2,1): 4"
                edge_part, cost_str = line.split(':')
                edge_part = edge_part.strip().strip('()')
                from_str, to_str = edge_part.split(',')
                from_id = int(from_str.strip())
                to_id = int(to_str.strip())
                cost = float(cost_str.strip())
                graph.add_edge(from_id, to_id, cost)

            elif section == "origin":
                # Format: "2"
                graph.origin = int(line.strip())

            elif section == "destinations":
                # Format: "5; 4" or "5;4" or "5"
                parts = line.split(';')
                graph.destinations = [int(p.strip()) for p in parts]

    return graph