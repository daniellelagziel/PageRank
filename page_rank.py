import matplotlib.pyplot as plt
import networkx as nx


def page_rank(g, d=0.7, pers_vector=None,
              max_iterations=100, err_tolerance=1.0e-6, starting_val=None, weight=None,
              dangling=None):
    """"
    !!!!!
    We used NetworkX - a Python package for the creation,
    manipulation, and study of the structure, dynamics, and functions of complex networks.
    !!!!!
    Parameters Explanation:
    !!!!!
    g(graph) :
      A NetworkX graph.
      Every Un-directed graph will be converted to a directed
      graph, with two directed edges for each un-directed edge.

    d(float) :
      Damping parameter for PageRank.
      Default value is 0.7.

    pers_vector(dictionary) :
      The pers_vector consisting of a dictionary with a
      key for every node in the graph
      and non-zero personalization value for each node.
      Default value is a uniform distribution.

    max_iterations(integer) :
      Maximum number of iterations in power method eigenvalue solver.
      Default value is 100.

    err_tolerance(float) :
      Error tolerance used to check convergence in power method solver.
      Default value is 1.0e-6.

    starting_val(dictionary) :
      Starting value of Page Rank iteration for each node in the graph.
      Default value is None.

    weight(key) :
      Edge data key to use as weight.
      If None weights are set to 1.

    dangling(dictionary) :
      The out edges to be assigned to any "dangling" nodes
      The dict key is the node the out edge points to and the dict
      value is the weight of that out edge.
      By default, dangling nodes are given
      out edges according to the pers vector (uniform if not specified).
      This must be selected to result in an irreducible transition
      matrix.
      It may be common to have the dangling dict
      to be the same as the pers vector.
    !!!!!
    Returns dictionary
    !!!!!
    Dictionary of nodes with PageRank as value.
    !!!!!

    """
    if len(g) == 0:
        return {}
    if not g.is_directed():
        directed_graph = g.to_directed()
    else:
        directed_graph = g
    w = nx.stochastic_graph(directed_graph, weight=weight)  # Create a copy in stochastic form
    n = w.number_of_nodes()
    if starting_val is None:
        x = dict.fromkeys(w, 1.0 / n)  # Choose fixed starting vector if not given
    else:
        s = float(sum(starting_val.values()))  # Normalized starting_val vector
        x = dict((k, v / s) for k, v in starting_val.items())
    if pers_vector is None:
        p = dict.fromkeys(w, 1.0 / n)  # Assign uniform vector if not given
    else:
        missing = set(g) - set(pers_vector)
        if missing:
            raise nx.NetworkXError('pers dictionary must have a value for every node. Missing nodes %s' % missing)
        s = float(sum(pers_vector.values()))
        p = dict((k, v / s) for k, v in pers_vector.items())
    if dangling is None:
        dangling_weights = p  # Use pers vector if dangling vector not specified
    else:
        missing = set(g) - set(dangling)
        if missing:
            raise nx.NetworkXError('Dangling node dictionary must have a value for every node. Missing nodes %s' % missing)
        s = float(sum(dangling.values()))
        dangling_weights = dict((k, v / s) for k, v in dangling.items())
    dangling_nodes = [n for n in w if w.out_degree(n, weight=weight) == 0.0]
    for _ in range(max_iterations):  # power iteration: make up to max_iter iterations
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = d * sum(xlast[n] for n in dangling_nodes)
        for n in x:
            for nbr in w[n]:  # this matrix multiply looks odd because it is doing a left multiply x^T=xlast^T*w
                x[nbr] += d * xlast[n] * w[n][nbr][weight]
            x[n] += danglesum * dangling_weights[n] + (1.0 - d) * p[n]
        err = sum([abs(x[n] - xlast[n]) for n in x])  # check convergence
        if err < n * err_tolerance:
            return x
    raise nx.NetworkXError('page rank: power iteration failed to converge in %d iterations.' % max_iterations)


G = nx.barabasi_albert_graph(30, 15)  # 30 is the number of nodes , 15 is the number of edges to attach from a new node to existing nodes
page_rank_values = page_rank(G, 0.7)  # 0.7 is the damping parameter
print("Page Rank Results:")
for pr in page_rank_values.values():
    print(pr)

nx.draw(G)  # This is our graph
plt.show()


"""
PageRank computes a ranking of the nodes in the graph G based on
    the structure of the incoming links. It was originally designed as
    an algorithm to rank web pages.
    Return the PageRank of the nodes in the graph.
    Notes
    -----
    The eigenvector calculation is done by the power iteration method
    and has no guarantee of convergence.  The iteration will stop
    after max_iter iterations or an error tolerance of
    number_of_nodes(G)*tol has been reached.

    The PageRank algorithm was designed for directed graphs but this
    algorithm does not check if the input graph is directed and will
    execute on undirected graphs by converting each edge in the
    directed graph to two edges.

    stochastic graph is a right-stochastic graph is a weighted digraph in which for each
    node, the sum of the weights of all the out-edges of that node is
    1. If the graph is already weighted (for example, via a 'weight'
    edge attribute), the reweighting takes that into account. """

