from panrep_model import PanRepHetero
from panrep_encoder import EncoderHGT
from panrep_decoders import NodeMotifDecoder,MultipleAttributeDecoder\
    ,MutualInformationDiscriminator,ClusterRecoverDecoderHomo
# need to import some modules

def train(model, optimizer, hetero_graph):
    model.train()
    optimizer.zero_grad()
    (loss, H) = model(hetero_graph) #heteoro_graph = g
    #we can return H to be the updated node embeddings
    loss.backward()
    optimizer.step()
    return loss.item()

# need to load graph first
# can adapt the following code from colab 5 to load heterogeneous graph
# TODO:: make check specific data structures and attributes of the graph
if 'IS_GRADESCOPE_ENV' not in os.environ:
    print("Device: {}".format(args['device']))

    # Load the data
    data = torch.load("acm.pkl")

    # Message types
    message_type_1 = ("paper", "author", "paper")
    message_type_2 = ("paper", "subject", "paper")

    # Dictionary of edge indices
    edge_index = {}
    edge_index[message_type_1] = data['pap']
    edge_index[message_type_2] = data['psp']

    # Dictionary of node features
    node_feature = {}
    node_feature["paper"] = data['feature']

    # Dictionary of node labels
    node_label = {}
    node_label["paper"] = data['label']

    # Load the train, validation and test indices
    train_idx = {"paper": data['train_idx'].to(args['device'])}
    val_idx = {"paper": data['val_idx'].to(args['device'])}
    test_idx = {"paper": data['test_idx'].to(args['device'])}

    # Construct a deepsnap tensor backend HeteroGraph
    hetero_graph = HeteroGraph(
        node_feature=node_feature,
        node_label=node_label,
        edge_index=edge_index,
        directed=True
    )

    print(f"ACM heterogeneous graph: {hetero_graph.num_nodes()} nodes, {hetero_graph.num_edges()} edges")

    # Node feature and node label to device
    for key in hetero_graph.node_feature:
        hetero_graph.node_feature[key] = hetero_graph.node_feature[key].to(args['device'])
    for key in hetero_graph.node_label:
        hetero_graph.node_label[key] = hetero_graph.node_label[key].to(args['device'])

    # Edge_index to sparse tensor and to device
    for key in hetero_graph.edge_index:
        edge_index = hetero_graph.edge_index[key]
        adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(hetero_graph.num_nodes('paper'), hetero_graph.num_nodes('paper')))
        hetero_graph.edge_index[key] = adj.t().to(args['device'])
    print(hetero_graph.edge_index[message_type_1])
    print(hetero_graph.edge_index[message_type_2])


# start training panrep
if 'IS_GRADESCOPE_ENV' not in os.environ:
    encoder = EncoderHGT(hetero_graph, ...) # fill in rest of parameters if using HGT encoder
    #otherwise use the HeteroGNN in colab 5 as encoder should be fine as well
    n_cluster, in_size, h_dim =
    decoders = {'mid':MutualInformationDiscriminator(),
            'crd':ClusterRecoverDecoderHomo(n_cluster, in_size, h_dim),
            'nmd':NodeMotifDecoder()}
    #specify parameters for decoder constructors above
    model = PanRepHetero(encoder, decoders).to(args['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    for epoch in range(args['epochs']):
        loss = train(model, optimizer, hetero_graph)