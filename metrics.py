from treelib import Node, Tree
from math import log
from itertools import permutations
from au import AnalysisUnit

def conciseness(tree, level, cs_name = None, result = None):
    '''We use the metric definition of simplicity from the supplement material from the paper
    "A Taxonomy of Evaluation Methods for Information Systems Artifacts" (Prat et al. 2015)'''

    depth_categories = 0
    depth_characteristics = 0
    number_categories = 0
    number_characteristics = 0

    root_node = tree.root

    for node in tree.all_nodes():
        if node.identifier == root_node:
            name = node.tag
            continue
        depth = tree.depth(node)
        if node.is_leaf():
            depth_characteristics += 1/depth
            number_characteristics += 1
        else:
            depth_categories += 1/depth
            number_categories += 1

    cc = 1/(1 + log((depth_categories + depth_characteristics) - 1))

    if level == 0:
        result = 'Classification system: '
    elif level == 1:
        result = result + '\tTable: '

    result = result + f'{name} | Number of categories/characteristics: {number_categories}/{number_characteristics} | Conciseness: {cc}\n'

    if level == 0:
        for table in tree.children("root"):
            subtree = tree.subtree(table.identifier)
            result = conciseness(subtree, level + 1, name, result)

    return result

def robustness(units):
    '''We determine robustness through:
    (a) the similarity of the nodes in a unit to each other
    (b) the dissimilarity of nodes in a unit to nodes outside the unit

    For (a), we take all pairs in a unit and calculate their similarity. The minimum similarity is our
    threshold against which we compare all nodes outside the unit (b). Note that (a) and (b) are calculated
    in the class AnalysisUnit, not here.

    The number of those "outside" nodes is the basis for our robustness metric. Basically, the lower the
    outside nodes in a unit, the higher the unit robustness.

    We normalize robustness so the value is in the interval [0,1], with 1 the highest possible robustness
    when there are no outside nodes.

    The overall robustness of a set of analysis units is their arithmetic mean.
    '''
    total_nodes = 0
    for unit in units:
        total_nodes = total_nodes + len(unit.nodes)

    rb = 0
    units_rb = []

    for unit in units:
        nodes_in_au = len(unit.nodes)
        outside_nodes = len(unit.outside_nodes)
        outside_proportion = outside_nodes / (nodes_in_au * (total_nodes - nodes_in_au))
        assert outside_proportion >= 0 and outside_proportion <= 1, f'Outside proportion is beyond expected interval: {outside_proportion}'
        rb = rb + 1 - outside_proportion
        units_rb.append((unit.identifier, nodes_in_au, outside_nodes, outside_proportion))

    result = f'Robustness: {rb / len(units)} | Units: {len(units)} | Total unit nodes: {total_nodes}\n'
    for unit_rb in sorted(units_rb, key=lambda tup: tup[3]):
        result = result + f'\tUnit: {unit_rb[0]} | Nodes: {unit_rb[1]} | Outside nodes: {unit_rb[2]} | Outside proportion: {unit_rb[3]}\n'

    return result


def create_analysis_units(tree, model):
    #We store the leaf nodes in a dictionary. Key is their parent node. Since it's a dictionary
    #multiple insertions of the same parent node with children has a performance impact, but we don't have to
    #take care of the filtering logic.
    leaf_node_units = dict()

    for node in tree.leaves():
        parent = tree.parent(node.identifier)
        children = tree.children(parent.identifier)

        for child in children:
            if child.is_leaf():
                if parent.identifier in leaf_node_units:
                    leaf_node_units[parent.identifier].add(child)
                else:
                    leaf_node_units[parent.identifier] = {child}

    analysis_units = list()
    for unit in leaf_node_units:
        nodes = list(leaf_node_units[unit])
        #Only with more than 2 nodes, we can create node pairs and calculate minimum and maximum similarity
        #Hence, analysis units with less than 3 nodes are not interesting.
        if len(nodes) > 2:
            id = tree.parent(nodes[0].identifier).identifier
            analysis_units.append(AnalysisUnit(id, nodes, model))

    for p in permutations(range(0, len(analysis_units)), 2):
        p_zero = analysis_units[p[0]]
        p_one = analysis_units[p[1]]
        p_zero.outside_similarity(p_one)

    return analysis_units
