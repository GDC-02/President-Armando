import matplotlib.pyplot as plt
import networkx as nx

def draw_neural_network_with_names():
    # layers layout
    layers = {
        'Input': 9,
        'HiddenLayer1': 5,  
        'HiddenLayer2': 3, 
        'Status': 1         #output
    }

    # input neuron names
    input_neuron_names = [
        "MDVP:Jitter(Abs)", "MDVP:PPQ", "MDVP:Shimmer", 
        "MDVP:APQ", "NHR", "RPDE", "DFA", "spread2", "PPE"
    ]

    G = nx.DiGraph()
    positions = {}
    labels = {}
    layer_x_positions = {}

    # layers position
    x_offset = 0
    for layer_name, num_nodes in layers.items():
        layer_x_positions[layer_name] = x_offset
        x_offset += 3  # Spacing between layers (increased for clarity)

    #nodes 
    y_spacing = 2.0  # Increased spacing between nodes
    for layer_name, num_nodes in layers.items():
        x = layer_x_positions[layer_name]
        for i in range(num_nodes):
            if layer_name == 'Input':
                # Use specific input neuron names
                node_name = input_neuron_names[i]
            elif layer_name == 'Status':
                # Rename the output neuron to "Status"
                node_name = "Status"
            else:
                # Use the custom names for the hidden layers
                node_name = f"{layer_name}_{i+1}"
            
            # Center output neuron and adjust spacing
            if layer_name == 'Status':
                y = 0  # Center the single output neuron
            else:
                y = -i * y_spacing + (num_nodes - 1) * y_spacing / 2
            
            G.add_node(node_name, layer=layer_name)
            positions[node_name] = (x, y)
            labels[node_name] = node_name  # Use the node name as the label

    # connections
    layer_names = list(layers.keys())
    for i in range(len(layer_names) - 1):
        current_layer = layer_names[i]
        next_layer = layer_names[i + 1]

        for current_node in range(layers[current_layer]):
            for next_node in range(layers[next_layer]):
                if current_layer == 'Input':
                    current_name = input_neuron_names[current_node]
                elif current_layer == 'Status':
                    current_name = "Status"
                else:
                    current_name = f"{current_layer}_{current_node+1}"
                
                next_name = f"{next_layer}_{next_node+1}" if next_layer != 'Status' else "Status"
                G.add_edge(current_name, next_name)


    plt.figure(figsize=(16, 12)) 
    nx.draw(
        G, 
        pos=positions, 
        with_labels=True, 
        labels=labels, 
        node_size=7600,  
        node_color='skyblue', 
        edge_color='black', 
        font_size=9, 
        font_weight='bold'
    )
    plt.title("President Armando", fontsize=24)
    plt.show()


draw_neural_network_with_names()
