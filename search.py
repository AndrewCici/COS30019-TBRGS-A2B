import math
from .graph import Graph
from .frontier import Frontier


def dfs_search(graph: Graph):
    start = graph.origin # Start node
    visited = set([start]) # Visited nodes
    frontier = Frontier('stack') # Container for dfs
    frontier.push((start, [start])) # Push start node to container
    nodes_created = 1 

    # Check if container is empty if not remove first node and return next
    while not frontier.is_empty():
        node_id, path = frontier.pop()


        # If node is destination return node
        if node_id in graph.destinations:
            return node_id, nodes_created, path

        # Retrieve neighbours' ids + cost in adjacency list -> neighbours = list of tuples
        neighbours = graph.adj.get(node_id, [])

        # Sort neighbours in ascending order
        sorted_neighbours = sorted(neighbours, key=lambda x: x[0])

        # Loop through reversed list cause LIFO
        for neighbour_id, edge_cost in reversed(sorted_neighbours):
                if neighbour_id not in visited:
                    visited.add(neighbour_id)
                    new_path = path + [neighbour_id] 
                    nodes_created += 1
                    frontier.push((neighbour_id, new_path))

    return None, nodes_created, None

def bfs_search(graph: Graph):
    start = graph.origin # Start node
    visited = set([start]) # Visited nodes
    frontier = Frontier('queue') # Container for dfs
    frontier.push((start, [start])) # Push start node to container
    nodes_created = 1 

    # Check if container is empty if not remove first node and return next
    while not frontier.is_empty():
        node_id, path = frontier.pop()

        # If node is destination return node
        if node_id in graph.destinations:
            return node_id, nodes_created, path

        # Retrieve neighbours' ids + cost in adjacency list -> neighbours = list of tuples
        neighbours = graph.adj.get(node_id, [])

        # Sort neighbours in ascending order
        sorted_neighbours = sorted(neighbours, key=lambda x: x[0])

        # Loop through normal list cause FIFO
        for neighbour_id, edge_cost in sorted_neighbours:
                if neighbour_id not in visited:
                    visited.add(neighbour_id)
                    new_path = path + [neighbour_id] 
                    nodes_created += 1
                    frontier.push((neighbour_id, new_path))
                    
    return None, nodes_created, None

# Đường chim bay
def heuristic(node_id: int, graph: Graph) -> float:
    if not graph.destinations:
        return 0.0
    node = graph.nodes[node_id]
    return min(math.sqrt((node.x - graph.nodes[d].x)**2 + (node.y - graph.nodes[d].y)**2) for d in graph.destinations)

# f = g + h
def a_star_search(graph: Graph):
    frontier = Frontier('priority')
    frontier.push((heuristic(graph.origin, graph), graph.origin, [graph.origin], 0.0)) #(f-value, start node, path, g cost)
    nodes_created = 1

    # Dictionary to store the best known g-cost to reach each node. Acts as a "closed list" that can be re-opened
    g_costs = {graph.origin: 0.0}

    while not frontier.is_empty():
        _, node_id, path, g_cost = frontier.pop() # Pop the smallest f value
        if node_id in graph.destinations:
            return node_id, nodes_created, path
        
        # If this entry's g-cost is worse than the best known cost for this node, skip it
        if g_cost > g_costs.get(node_id, float('inf')):
            continue
        
        # Generate all neighbors of the current node.
        for n_id, cost in graph.adj.get(node_id, []):
            # Calculate the total cost to reach this neighbor via the current path.
            new_g = g_cost + cost

            # If this neighbor has never been visited, OR we have found a cheaper path to it, update its g-cost and push it.
            if n_id not in g_costs or new_g < g_costs[n_id]:
                # Record the new best cost.
                g_costs[n_id] = new_g

                # Increment the node counter because we are creating a new search node.
                nodes_created += 1

                # Push the neighbor onto the frontier with its new f-value.
                frontier.push((
                    new_g + heuristic(n_id, graph),   # f(n) = g(n) + h(n)
                    n_id,                             # node ID
                    path + [n_id],                    # updated path
                    new_g                             # accumulated g-cost
                ))
                    
    return None, nodes_created, None

# Tương tự heuristic.
def gbfs_search(graph: Graph):
    frontier = Frontier('priority')
    frontier.push((heuristic(graph.origin, graph), graph.origin, [graph.origin]))
    explored = set()
    nodes_created = 1

    while not frontier.is_empty():
        _, node_id, path = frontier.pop()
        if node_id in graph.destinations:
            return node_id, nodes_created, path
            
        if node_id not in explored:
            explored.add(node_id)
            for n_id, _ in graph.adj.get(node_id, []):
                if n_id not in explored:
                    nodes_created += 1
                    frontier.push((heuristic(n_id, graph), n_id, path + [n_id]))
                    
    return None, nodes_created, None

# Là A* nhưng h = 0, sử dụng g(cost) để mở rộng nút.
def ucs_search(graph: Graph):
    frontier = Frontier('priority')
    frontier.push((0.0, graph.origin, [graph.origin]))
    explored = set()
    nodes_created = 1

    while not frontier.is_empty():
        g_cost, node_id, path = frontier.pop()
        if node_id in graph.destinations:
            return node_id, nodes_created, path
            
        if node_id not in explored:
            explored.add(node_id)
            for n_id, cost in graph.adj.get(node_id, []):
                if n_id not in explored:
                    nodes_created += 1
                    frontier.push((g_cost + cost, n_id, path + [n_id]))
                    
    return None, nodes_created, None

def ida_star_search(graph: Graph):
    # Khởi tạo giới hạn ban đầu là h(origin)
    bound = heuristic(graph.origin, graph)
    nodes_created = 1

    def search(node_id, g_cost, current_bound, current_visited, current_path):
        nonlocal nodes_created
        f_cost = g_cost + heuristic(node_id, graph)
        
        # Nếu f_cost vượt giới hạn hiện tại, trả về f_cost này để làm giới hạn cho vòng sau
        if f_cost > current_bound:
            return f_cost, None, None
            
        if node_id in graph.destinations:
            return "FOUND", node_id, current_path

        min_bound = float('inf')
        
        # Lấy và sắp xếp các đỉnh kề (Tie-breaking: ID nhỏ hơn duyệt trước)
        neighbours = graph.adj.get(node_id, [])
        for n_id, edge_cost in sorted(neighbours, key=lambda x: x[0]):
            if n_id not in current_visited:
                current_visited.add(n_id)
                nodes_created += 1
                
                # Gọi đệ quy xuống nhánh sâu hơn
                res_bound, res_id, res_path = search(n_id, g_cost + edge_cost, current_bound, current_visited, current_path + [n_id])
                
                if res_bound == "FOUND":
                    return "FOUND", res_id, res_path
                if res_bound < min_bound:
                    min_bound = res_bound
                    
                # Backtrack: Xóa node khỏi path hiện tại để cho phép các nhánh khác đi qua
                current_visited.remove(n_id)
                
        return min_bound, None, None

    # Vòng lặp tăng dần giới hạn (Iterative Deepening)
    while True:
        res_bound, goal_id, goal_path = search(graph.origin, 0.0, bound, {graph.origin}, [graph.origin])
        if res_bound == "FOUND":
            return goal_id, nodes_created, goal_path
        if res_bound == float('inf'): # Không thể đi tới đích
            return None, nodes_created, None
        bound = res_bound # Cập nhật giới hạn mới cho vòng lặp tiếp theo
