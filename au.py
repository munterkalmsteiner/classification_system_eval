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
            self._similarity = similarity
            self.unknown_tokens = unknown_tokens

        def __eq__(self, other):
            return self.similarity() == other.similarity()

        def __lt__(self, other):
            return self.similarity() < other.similarity()

        def similarity(self):
            if self.has_unknown_tokens():
                return -1
            else:
                return self._similarity

        def has_unknown_tokens(self):
            return len(self.unknown_tokens) > 0

        def describe(self):
            return f'Pair: {self.node0.tag} / {self.node1.tag} | {self.__describe_similarity()}'

        def __describe_similarity(self):
            ut = None
            if self.has_unknown_tokens():
                ut = f'Unknown tokens: {" ".join(self.unknown_tokens)} / '

            return f'{ut}Similarity: {self.similarity()}'

    def __init__(self, identifier, nodes, model):
        self.nodes = nodes
        self.unusable_nodes = set()
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

            if len(tokens0) == 0:
                self.unusable_nodes.add(leaf0)
                continue

            if len(tokens1) == 0:
                self.unusable_nodes.add(leaf1)
                continue

            unknown_tokens0 = self.__find_unknown_tokens(tokens0)
            unknown_tokens1 = self.__find_unknown_tokens(tokens1)

            if len(unknown_tokens0) != 0:
                leaf0.data = unknown_tokens0
                self.unusable_nodes.add(leaf0)
                continue

            if len(unknown_tokens1) != 0:
                leaf1.data = unknown_tokens1
                self.unusable_nodes.add(leaf1)
                continue

            similarity = self.doc2vec.wv.n_similarity(tokens0, tokens1)
            self.pairs.append(self.Pair(leaf0, leaf1, similarity))


    def __eq__(self, other):
        return (self.minimum_similarity, self.min_max_similarity()) == (other.minimum_similarity, other.min_max_similarity())

    def __lt__(self, other):
        return (self.minimum_similarity, self.min_max_similarity()) < (other.minimum_similarity, other.min_max_similarity())

    def outside_similarity(self, other):
        """This is the key measure for defining robustness. Here we measure for each node the similarity to each node from the other analysis unit. If the similarity to an other node is higher than the minimum similarity within the analysis unit, we add that outside node to a list in this analysis unit. Note that we can calculate the outside similarity if we have a minimum similarity, i.e. we have more than one pair in this analysis unit."""
        if len(self.pairs) > 1:
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
        if self.min_similarity == -1 or self.max_similarity == -1:
            return -1
        else:
            return self.maximum_similarity - self.minimum_similarity

    def __edge_values_similarity(self, rev):
        value = -1

        if len(self.pairs) > 1:
            sorted_pairs = sorted(self.pairs, reverse = rev)
            value = sorted_pairs[0].similarity()

        return value

    def describe(self):
        description = f'Identifier: {self.identifier}\nNumber of nodes: {len(self.nodes)}\nMin|Max similarity: {self.minimum_similarity}|{self.maximum_similarity}\nMin-Max similarity: {self.min_max_similarity()}\nNumber of pairs: {len(self.pairs)}\n'
        for node in self.unusable_nodes:
            description = description + f'\tIdentifier: {node.identifier} | Content: {node.tag} | Unknown tokens: {node.data}\n'
        description = description + f'Number of pairs: {len(self.pairs)}\n'
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
