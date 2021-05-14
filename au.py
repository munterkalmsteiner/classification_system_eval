from gensim.models.doc2vec import Doc2Vec
from gensim.utils import simple_preprocess
from functools import total_ordering
from itertools import combinations
from cached_property import cached_property
from treelib import Node, Tree

@total_ordering
class AnalysisUnit:
    @total_ordering
    class Pair:
        def __init__(self, node0, node1, similarity, unknown_tokens = []):
            self.node0 = node0
            self.node1 = node1
            self.similarity = similarity
            self.unknown_tokens = unknown_tokens

        def __eq__(self, other):
            return self.similarity == other.similarity

        def __lt__(self, other):
            return self.similarity < other.similarity

        def has_unknown_tokens(self):
            return len(self.unknown_tokens) > 0

        def describe(self):
            return f'Pair: {self.node0.tag} / {self.node1.tag} | {self.__describe_similarity()}'

        def __describe_similarity(self):
            if self.has_unknown_tokens():
                return f'Unknown tokens: {" ".join(self.unknown_tokens)}'
            else:
                return f'Similarity: {self.similarity}'

    def __init__(self, identifier, nodes, model):
        self.nodes = nodes
        self.doc2vec = model
        self.pairs = list()
        self.identifier = identifier
        self.outside_nodes = list()
        self.min_similarity = -1
        self.max_similarity = -1

        for pair in combinations(range(0, len(self.nodes)), 2):
            leaf0 = self.nodes[pair[0]]
            leaf1 = self.nodes[pair[1]]
            tokens0 = simple_preprocess(leaf0.tag, max_len=100)
            tokens1 = simple_preprocess(leaf1.tag, max_len=100)

            unknown_tokens = self.__find_unknown_tokens(tokens0 + tokens1)

            if len(tokens0) == 0 or len(tokens1) == 0:
                self.pairs.append(self.Pair(leaf0, leaf1, -1, [leaf0.tag, leaf1.tag]))
                continue

            if len(unknown_tokens) != 0:
                self.pairs.append(self.Pair(leaf0, leaf1, -1, unknown_tokens))
                continue

            similarity = self.doc2vec.wv.n_similarity(tokens0, tokens1)
            self.pairs.append(self.Pair(leaf0, leaf1, similarity))


    def __eq__(self, other):
        return (self.minimum_similarity, self.min_max_similarity()) == (other.minimum_similarity, other.min_max_similarity())

    def __lt__(self, other):
        return (self.minimum_similarity, self.min_max_similarity()) < (other.minimum_similarity, other.min_max_similarity())

    def outside_similarity(self, other):
        """This is the key measure for defining robustness. Here we measure for each node the similarity to each node from the other analysis unit. If the similarity to an other node is higher than the minimum similarity within the analysis unit, we add that outside node to a list in this analysis unit. Note that we can calculate the outside similarity if we have a minimum similarity, i.e. we have more than two nodes in this analysis unit."""
        if len(self.nodes) > 2:
            for self_node in self.nodes:
                for other_node in other.nodes:
                    self_tokens = simple_preprocess(self_node.tag, max_len=100)
                    other_tokens = simple_preprocess(other_node.tag, max_len=100)

                    unknown_tokens = self.__find_unknown_tokens(self_tokens + other_tokens)

                    if len(unknown_tokens) == 0 and len(self_tokens) > 0 and len(other_tokens) > 0:
                        similarity = self.doc2vec.wv.n_similarity(self_tokens, other_tokens)
                        if similarity > self.minimum_similarity:
                            self.outside_nodes.append((self_node, other_node, similarity))

    @cached_property
    def minimum_similarity(self):
        self.min_similarity = self.__edge_values_similarity(False)
        return self.min_similarity

    @cached_property
    def maximum_similarity(self):
        self.max_similarity = self.__edge_values_similarity(True)
        return self.max_similarity

    def min_max_similarity(self):
        minimum = self.minimum_similarity
        maximum = self.maximum_similarity
        if minimum != maximum:
            return maximum - minimum
        else:
            return maximum

    def __edge_values_similarity(self, rev):
        value = -1

        for pair in sorted(self.pairs, reverse = rev):
            if pair.has_unknown_tokens():
                continue
            else:
                value = pair.similarity
                break

        if len(self.pairs) == 1 and value != -1:
            value = 0

        return value

    def describe(self):
        description = f'Number of nodes: {len(self.nodes)}\nIdentifier: {self.identifier}\nMin|Max similarity: {self.minimum_similarity}|{self.maximum_similarity}\nMin-Max similarity: {self.min_max_similarity()}\n'
        for pair in sorted(self.pairs, reverse = True):
            description = description + f'\t{pair.describe()}\n'
        description = description + f'Number of similar outside nodes: {len(self.outside_nodes)}\n'
        for outside_node in self.outside_nodes:
            description = description + f'\tI: {outside_node[0]} | O: {outside_node[1]} | S: {outside_node[2]}\n'
        return description

    def __find_unknown_tokens(self, tokens):
        result = list()
        for token in tokens:
            if not token in self.doc2vec.wv.vocab:
                result.append(token)
        return result
