from flask import Flask, render_template, request, jsonify
import networkx as nx
import random
import json

app = Flask(__name__)

# 7 thuật toán sampling
def sample_random_node(G, qn, seed=0):
    """Random Node (RN): Chọn ngẫu nhiên các node"""
    rnd = random.Random(seed)
    nodes = list(G.nodes())
    selected = rnd.sample(nodes, min(qn, len(nodes)))
    steps = [{"node": n, "type": "select", "desc": f"Chọn node {n}"} for n in selected]
    return G.subgraph(selected).copy(), steps

def sample_random_edge(G, qn, seed=0):
    """Random Edge (RE): Chọn ngẫu nhiên các cạnh và lấy 2 đầu"""
    rnd = random.Random(seed)
    edges = list(G.edges())
    if not edges:
        return sample_random_node(G, qn, seed)
    
    selected = set()
    steps = []
    attempts = 0
    target = min(qn, G.number_of_nodes())
    
    while len(selected) < target and attempts < 10000:
        u, v = rnd.choice(edges)
        # Kiểm tra cẩn thận để không vượt qn
        if u not in selected and len(selected) < target:
            selected.add(u)
            steps.append({"node": u, "edge": (u, v), "type": "edge", "desc": f"Chọn cạnh ({u},{v}), thêm node {u}"})
        if v not in selected and len(selected) < target:
            selected.add(v)
            steps.append({"node": v, "edge": (u, v), "type": "edge", "desc": f"Thêm node {v} từ cạnh ({u},{v})"})
        attempts += 1
    
    # Nếu thiếu nodes, fill random từ remaining
    if len(selected) < target:
        remaining = list(set(G.nodes()) - selected)
        rnd.shuffle(remaining)
        need = target - len(selected)
        for node in remaining[:need]:
            selected.add(node)
            steps.append({"node": node, "type": "fill", "desc": f"Thêm node {node} (fill còn thiếu)"})
    
    return G.subgraph(selected).copy(), steps

def sample_degree_node(G, qn, seed=0):
    """Degree Node (DN): Chọn node theo xác suất dựa trên degree (WITHOUT replacement).
    Fix chắc chắn:
    - Không bao giờ lỗi khi có nhiều node degree=0
    - Luôn trả về đúng qn node (hoặc min(qn, |V|))
    - Nếu thiếu node degree>0 thì fill từ node degree=0 (uniform)
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    nodes = list(G.nodes())
    target = min(int(qn), len(nodes))
    if target <= 0:
        return G.subgraph([]).copy(), []

    degrees = {n: G.degree(n) for n in nodes}

    # Tách 2 nhóm: degree>0 và degree=0
    pos_nodes = [n for n in nodes if degrees[n] > 0]
    zero_nodes = [n for n in nodes if degrees[n] == 0]

    selected = []
    steps = []

    # 1) Chọn trong nhóm degree>0 theo weight (without replacement)
    if len(pos_nodes) > 0:
        w = np.array([degrees[n] for n in pos_nodes], dtype=float)
        w_sum = float(w.sum())

        # safety
        if w_sum > 0:
            p = w / w_sum
            k = min(target, len(pos_nodes))
            pick = rng.choice(np.array(pos_nodes, dtype=object), size=k, replace=False, p=p)
            for node in pick:
                node = int(node)
                selected.append(node)
                steps.append({
                    "node": node,
                    "degree": int(degrees[node]),
                    "type": "degree",
                    "desc": f"Chọn node {node} (degree={degrees[node]})"
                })

    # 2) Nếu chưa đủ target, fill từ các node còn lại (ưu tiên degree=0) — uniform
    need = target - len(selected)
    if need > 0:
        remaining = [n for n in nodes if n not in set(selected)]

        # ưu tiên fill từ nhóm degree=0 trước (nhìn hợp lý & tránh bias thêm)
        fill_pool = [n for n in zero_nodes if n in remaining]
        if len(fill_pool) < need:
            # nếu vẫn thiếu, lấy thêm từ remaining bất kỳ
            fill_pool = remaining

        fill_k = min(need, len(fill_pool))
        fill_pick = rng.choice(np.array(fill_pool, dtype=object), size=fill_k, replace=False)

        for node in fill_pick:
            node = int(node)
            if node in set(selected):
                continue
            selected.append(node)
            steps.append({
                "node": node,
                "degree": int(degrees[node]),
                "type": "fill",
                "desc": f"Thêm node {node} (fill còn thiếu, degree={degrees[node]})"
            })

    return G.subgraph(selected).copy(), steps

def sample_bfs(G, qn, seed=0):
    """BFS: Duyệt theo chiều rộng từ một node ngẫu nhiên"""
    rnd = random.Random(seed)
    start = rnd.choice(list(G.nodes()))
    visited = {start}
    queue = [start]
    steps = [{"node": start, "type": "start", "desc": f"Bắt đầu từ node {start}"}]
    target = min(qn, G.number_of_nodes())
    
    while queue and len(visited) < target:
        node = queue.pop(0)
        for neighbor in list(G.neighbors(node)):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                steps.append({"node": neighbor, "type": "visit", "desc": f"Thăm node {neighbor}"})
                if len(visited) >= target:
                    break
    
    # Nếu không đủ nodes (đồ thị không liên thông), thêm random nodes còn lại
    if len(visited) < target:
        remaining = list(set(G.nodes()) - visited)
        rnd.shuffle(remaining)
        need = target - len(visited)
        for node in remaining[:need]:
            visited.add(node)
            steps.append({"node": node, "type": "fill", "desc": f"Thêm node {node} (đồ thị không liên thông)"})
    
    return G.subgraph(visited).copy(), steps

def sample_dfs(G, qn, seed=0):
    """DFS: Duyệt theo chiều sâu từ một node ngẫu nhiên"""
    rnd = random.Random(seed)
    start = rnd.choice(list(G.nodes()))
    visited = set()
    stack = [start]
    steps = [{"node": start, "type": "start", "desc": f"Bắt đầu từ node {start}"}]
    target = min(qn, G.number_of_nodes())
    
    while stack and len(visited) < target:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        if node != start:
            steps.append({"node": node, "type": "visit", "desc": f"Thăm node {node}"})
        
        neighbors = list(G.neighbors(node))
        rnd.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in visited:
                stack.append(neighbor)
                steps.append({"node": neighbor, "type": "stack", "desc": f"Thêm node {neighbor} vào stack"})
    
    # Nếu không đủ nodes (đồ thị không liên thông), thêm random nodes còn lại
    if len(visited) < target:
        remaining = list(set(G.nodes()) - visited)
        rnd.shuffle(remaining)
        need = target - len(visited)
        for node in remaining[:need]:
            visited.add(node)
            steps.append({"node": node, "type": "fill", "desc": f"Thêm node {node} (đồ thị không liên thông)"})
    
    return G.subgraph(visited).copy(), steps

def sample_random_walk(G, qn, seed=0):
    """Random Walk (RW): Đi bộ ngẫu nhiên trên đồ thị
    Bản này đảm bảo: LUÔN trả về đúng min(qn, |V|) node.
    """
    rnd = random.Random(seed)
    nodes_list = list(G.nodes())
    target = min(int(qn), G.number_of_nodes())

    if target <= 0 or not nodes_list:
        return G.subgraph([]).copy(), []

    current = rnd.choice(nodes_list)
    visited = set([current])
    steps = [{"node": current, "type": "start", "desc": f"Bắt đầu tại node {current}"}]

    # tăng nhẹ budget để ít khi phải fill, nhưng vẫn giới hạn để không chạy lâu
    max_steps = min(max(target * 25, target * 10), 50000)
    step_count = 0

    while len(visited) < target and step_count < max_steps:
        neighbors = list(G.neighbors(current))
        if not neighbors:
            current = rnd.choice(nodes_list)
            steps.append({"node": current, "type": "restart", "desc": f"Restart tại node {current}"})
        else:
            next_node = rnd.choice(neighbors)
            if next_node not in visited:
                steps.append({"node": next_node, "from": current, "type": "walk",
                              "desc": f"Đi từ {current} → {next_node}"})
            current = next_node
            visited.add(current)

        step_count += 1
        if step_count % 5000 == 0 and len(visited) < target:
            current = rnd.choice(nodes_list)

    # Nếu vẫn thiếu unique nodes, fill cho đủ target (để demo luôn “đúng qn”)
    if len(visited) < target:
        remaining = list(set(G.nodes()) - visited)
        rnd.shuffle(remaining)
        need = target - len(visited)
        for node in remaining[:need]:
            visited.add(node)
            steps.append({"node": node, "type": "fill", "desc": f"Thêm node {node} (fill còn thiếu)"})

    return G.subgraph(visited).copy(), steps

def sample_random_node_neighbor(G, qn, seed=0):
    """Random Node Neighbor (RNN): Chọn node và các hàng xóm của nó"""
    rnd = random.Random(seed)
    nodes = list(G.nodes())
    selected = set()
    steps = []
    attempts = 0
    target = min(qn, G.number_of_nodes())
    
    while len(selected) < target and attempts < 10000:
        node = rnd.choice(nodes)
        # Chỉ add root nếu còn chỗ
        if node not in selected and len(selected) < target:
            selected.add(node)
            steps.append({"node": node, "type": "select", "desc": f"Chọn node {node}"})
        
        # Add neighbors, dừng đúng khi đủ qn (không vượt)
        for neighbor in G.neighbors(node):
            if neighbor not in selected and len(selected) < target:
                selected.add(neighbor)
                steps.append({"node": neighbor, "type": "neighbor", 
                             "desc": f"Thêm hàng xóm {neighbor} của node {node}"})
            if len(selected) >= target:
                break
        attempts += 1
    
    # Nếu thiếu nodes, fill random từ remaining
    if len(selected) < target:
        remaining = list(set(G.nodes()) - selected)
        rnd.shuffle(remaining)
        need = target - len(selected)
        for node in remaining[:need]:
            selected.add(node)
            steps.append({"node": node, "type": "fill", "desc": f"Thêm node {node} (fill còn thiếu)"})
    
    return G.subgraph(selected).copy(), steps

SAMPLERS = {
    "RN": {"name": "Random Node", "func": sample_random_node, "desc": "Chọn ngẫu nhiên các node"},
    "RE": {"name": "Random Edge", "func": sample_random_edge, "desc": "Chọn ngẫu nhiên các cạnh"},
    "DN": {"name": "Degree Node", "func": sample_degree_node, "desc": "Chọn node theo độ cao (degree)"},
    "BFS": {"name": "Breadth-First Search", "func": sample_bfs, "desc": "Duyệt theo chiều rộng"},
    "DFS": {"name": "Depth-First Search", "func": sample_dfs, "desc": "Duyệt theo chiều sâu"},
    "RW": {"name": "Random Walk", "func": sample_random_walk, "desc": "Đi bộ ngẫu nhiên"},
    "RNN": {"name": "Random Node-Neighbor", "func": sample_random_node_neighbor, "desc": "Chọn node và các hàng xóm"},
}

def generate_graph(n=30, p=0.15, seed=0):
    """Tạo đồ thị ngẫu nhiên đơn giản"""
    random.seed(seed)
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    # Đảm bảo đồ thị liên thông
    if not nx.is_connected(G):
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()
    return G

def generate_community_graph(n=30, seed=0):
    """Tạo đồ thị với 2 communities rõ ràng"""
    random.seed(seed)
    # Chia làm 2 communities
    n1 = n // 2
    n2 = n - n1
    
    # Tạo 2 communities với kết nối nội bộ cao
    G = nx.Graph()
    
    # Community 1: nodes 0 to n1-1
    for i in range(n1):
        G.add_node(i, community=0)
    # Kết nối trong community 1 (p_in = 0.3)
    for i in range(n1):
        for j in range(i+1, n1):
            if random.random() < 0.3:
                G.add_edge(i, j)
    
    # Community 2: nodes n1 to n-1
    for i in range(n1, n):
        G.add_node(i, community=1)
    # Kết nối trong community 2 (p_in = 0.3)
    for i in range(n1, n):
        for j in range(i+1, n):
            if random.random() < 0.3:
                G.add_edge(i, j)
    
    # Kết nối giữa 2 communities (p_out = 0.03)
    for i in range(n1):
        for j in range(n1, n):
            if random.random() < 0.03:
                G.add_edge(i, j)
    
    return G

def generate_core_periphery_graph(n=30, seed=0):
    """Tạo đồ thị Core-Periphery"""
    random.seed(seed)
    n_core = n // 4  # 25% là core
    n_periphery = n - n_core
    
    G = nx.Graph()
    
    # Core nodes: 0 to n_core-1
    for i in range(n_core):
        G.add_node(i, group='core')
    # Kết nối dày đặc trong core (p = 0.6)
    for i in range(n_core):
        for j in range(i+1, n_core):
            if random.random() < 0.6:
                G.add_edge(i, j)
    
    # Periphery nodes: n_core to n-1
    for i in range(n_core, n):
        G.add_node(i, group='periphery')
    
    # Kết nối periphery với core (mỗi periphery node kết nối với 1-2 core nodes)
    for i in range(n_core, n):
        num_connections = random.randint(1, 2)
        core_nodes = random.sample(range(n_core), num_connections)
        for core_node in core_nodes:
            G.add_edge(i, core_node)
    
    # Ít kết nối giữa các periphery nodes (p = 0.05)
    for i in range(n_core, n):
        for j in range(i+1, n):
            if random.random() < 0.05:
                G.add_edge(i, j)
    
    return G

@app.route("/")
def index():
    return render_template("demo.html", samplers=SAMPLERS)

@app.route("/sample", methods=["POST"])
def sample():
    data = request.json
    method = data.get("method", "RN")
    graph_type = data.get("graph_type", "random")
    n = int(data.get("n", 30))
    qn = int(data.get("qn", 15))
    seed = int(data.get("seed", 0))
    
    # Tạo đồ thị theo loại
    if graph_type == "community":
        G = generate_community_graph(n=n, seed=seed)
    elif graph_type == "core_periphery":
        G = generate_core_periphery_graph(n=n, seed=seed)
    else:
        G = generate_graph(n=n, p=0.15, seed=seed)
    
    # Áp dụng thuật toán sampling
    sampler = SAMPLERS[method]["func"]
    subgraph, steps = sampler(G, qn, seed)
    
    # Chuyển đổi sang format JSON với thông tin nhóm
    nodes_data = []
    for node in G.nodes():
        node_info = {"id": node, "label": str(node)}
        if graph_type == "community" and 'community' in G.nodes[node]:
            node_info["group"] = f"comm{G.nodes[node]['community']}"
        elif graph_type == "core_periphery" and 'group' in G.nodes[node]:
            node_info["group"] = G.nodes[node]['group']
        nodes_data.append(node_info)
    
    graph_data = {
        "nodes": nodes_data,
        "edges": [{"source": u, "target": v} for u, v in G.edges()]
    }
    
    subgraph_data = {
        "nodes": list(subgraph.nodes()),
        "edges": list(subgraph.edges())
    }
    
    return jsonify({
        "graph": graph_data,
        "subgraph": subgraph_data,
        "steps": steps,
        "graph_type": graph_type
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
