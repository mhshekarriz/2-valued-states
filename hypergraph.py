from typing import List, Set, Tuple

class Hypergraph:
    def __init__(self, num_vertices: int, hyperedges: List[List[int]]):
        self.num_vertices = num_vertices
        self.hyperedges = hyperedges
        self.uniform_size = len(hyperedges[0]) if hyperedges else 0
        for he in hyperedges:
            if len(he) != self.uniform_size:
                raise ValueError("All hyperedges must be of the same size (n-uniform).")

    def generate_valid_states(self) -> List[List[int]]:
        """Generate all valid 2-valued states: exactly one 1 per hyperedge, others 0, consistently."""
        valid_states: Set[Tuple[int, ...]] = set()
        initial_state = [-1] * self.num_vertices

        def backtrack(hyperedge_index: int, current_state: List[int]):
            if hyperedge_index == len(self.hyperedges):
                if all(bit != -1 for bit in current_state):
                    valid_states.add(tuple(current_state))
                return

            hyperedge = self.hyperedges[hyperedge_index]
            seen_in_hyperedge = [v - 1 for v in hyperedge if current_state[v - 1] != -1]
            ones_in_seen = sum(1 for idx in seen_in_hyperedge if current_state[idx] == 1)
            unseen_in_hyperedge = [v - 1 for v in hyperedge if current_state[v - 1] == -1]

            if ones_in_seen > 1:
                return  # Backtrack: invalid state in this path

            if ones_in_seen == 1:
                # Assign 0 to all unseen vertices in the current hyperedge
                next_state = list(current_state)
                for idx in unseen_in_hyperedge:
                    next_state[idx] = 0
                backtrack(hyperedge_index + 1, next_state)
            elif ones_in_seen == 0:
                # Try assigning 1 to each unseen vertex one by one
                for i, unseen_idx in enumerate(unseen_in_hyperedge):
                    temp_state = list(current_state)
                    temp_state[unseen_idx] = 1
                    for other_unseen_idx in unseen_in_hyperedge:
                        if other_unseen_idx != unseen_idx:
                            temp_state[other_unseen_idx] = 0
                    backtrack(hyperedge_index + 1, temp_state)
            else:
                pass

        backtrack(0, initial_state)
        return [list(state) for state in valid_states]

    def are_adjacent(self, u: int, v: int) -> bool:
        """Checks if two vertices are adjacent (belong to the same hyperedge)."""
        for hyperedge in self.hyperedges:
            if u in hyperedge and v in hyperedge:
                return True
        return False

    def is_tifs(self, states: List[List[int]]) -> List[Tuple[int, int]]:
        """
        Determines if there are non-adjacent vertices u and v (with u < v) such that whenever u is 1, v is 0
        and u is 1 in at least one state.
        Returns a list of ordered pairs (u, v) satisfying this condition.
        """
        tifs_pairs = []
        for u in range(1, self.num_vertices + 1):
            for v in range(u + 1, self.num_vertices + 1):  # Ensure u < v
                if not self.are_adjacent(u, v):
                    u_is_one_in_some_state = any(state[u - 1] == 1 for state in states)
                    if u_is_one_in_some_state:
                        implies_not = all(state[u - 1] == 1 and state[v - 1] == 0 for state in states if state[u - 1] == 1)
                        if implies_not:
                            tifs_pairs.append((u, v))
        return tifs_pairs

    def is_tits(self, states: List[List[int]]) -> List[Tuple[int, int]]:
        """
        Determines if there are non-adjacent vertices u and v (with u < v) such that whenever u is 1, v is also 1
        and u is 1 in at least one state.
        Returns a list of ordered pairs (u, v) satisfying this condition.
        """
        tits_pairs = []
        for u in range(1, self.num_vertices + 1):
            for v in range(u + 1, self.num_vertices + 1):  # Ensure u < v
                if not self.are_adjacent(u, v):
                    u_is_one_in_some_state = any(state[u - 1] == 1 for state in states)
                    if u_is_one_in_some_state:
                        implies = all(state[u - 1] == 1 and state[v - 1] == 1 for state in states if state[u - 1] == 1)
                        if implies:
                            tits_pairs.append((u, v))
        return tits_pairs

    def is_not(self, states: List[List[int]]) -> List[Tuple[int, int]]:
            """
            Determines if there are non-adjacent vertices u and v such that whenever u is 1, v is 0,
            and whenever u is 0, v is 1 (and vice versa), and both u and v have both 0 and 1
            in at least one state each.
            Returns a list of unordered pairs {u, v} satisfying this.
            """
            not_pairs = []
            for u in range(1, self.num_vertices + 1):
                for v in range(u + 1, self.num_vertices + 1):  # Check unordered pairs
                    if not self.are_adjacent(u, v):
                        u_has_one = any(state[u - 1] == 1 for state in states)
                        u_has_zero = any(state[u - 1] == 0 for state in states)
                        v_has_one = any(state[v - 1] == 1 for state in states)
                        v_has_zero = any(state[v - 1] == 0 for state in states)

                        if u_has_one and u_has_zero and v_has_one and v_has_zero:
                            u_implies_not_v = all(state[u - 1] == 1 and state[v - 1] == 0 for state in states if state[u - 1] == 1)
                            u_zero_implies_v_one = all(state[u - 1] == 0 and state[v - 1] == 1 for state in states if state[u - 1] == 0)
                            v_implies_not_u = all(state[v - 1] == 1 and state[u - 1] == 0 for state in states if state[v - 1] == 1)
                            v_zero_implies_u_one = all(state[v - 1] == 0 and state[u - 1] == 1 for state in states if state[v - 1] == 0)

                            if u_implies_not_v and u_zero_implies_v_one and v_implies_not_u and v_zero_implies_u_one:
                                not_pairs.append(tuple(sorted((u, v))))  # Store as sorted tuple for uniqueness
            return list(set(not_pairs)) # Remove duplicates if any

    def is_and(self, states: List[List[int]]) -> List[Tuple[int, int, int]]:
        """
        Determines if there are non-adjacent vertices u, v, and w such that for every state s,
        the value of w is the AND of the values of u and v.
        Returns a list of ordered triples (u, v, w) satisfying this.
        """
        and_triples = []
        for u in range(1, self.num_vertices + 1):
            for v in range(u + 1, self.num_vertices + 1):
                if not self.are_adjacent(u, v):
                    for w in range(1, self.num_vertices + 1):
                        if w != u and w != v and not self.are_adjacent(u, w) and not self.are_adjacent(v, w):
                            is_and_relation = True
                            for state in states:
                                and_value = state[u - 1] and state[v - 1]
                                if state[w - 1] != and_value:
                                    is_and_relation = False
                                    break
                            if is_and_relation:
                                and_triples.append(tuple(sorted((u, v)) + [w])) # Sort u, v for consistency
        return list(set(and_triples)) # Remove duplicates

    def is_unital(self, states: List[List[int]]) -> bool:
        """Unital: each vertex has value 1 in at least one state."""
        for i in range(self.num_vertices):
            if not any(state[i] == 1 for state in states):
                return False
        return True

    def is_separating(self, states: List[List[int]]) -> bool:
        """Separating: every pair of distinct vertices is distinguished by some state."""
        for i in range(self.num_vertices):
            for j in range(i + 1, self.num_vertices):
                if all(state[i] == state[j] for state in states):
                    return False
        return True

    def analyze(self):
        print(f"Number of vertices: {self.num_vertices}")
        print(f"Hyperedges: {self.hyperedges}")
        print()

        states = self.generate_valid_states()
        print(f"Number of valid 2-valued states: {len(states)}")
        for idx, state in enumerate(states):
            print(f"State {idx + 1}: {' '.join(str(bit) for bit in state)}")

        print("\nUnital:", self.is_unital(states))
        print("Separating:", self.is_separating(states))

        tifs_pairs = self.is_tifs(states)
        if tifs_pairs:
            print("TIFS pairs (u -> not v) where u < v:", tifs_pairs)
        else:
            print("No TIFS pairs found.")

        tits_pairs = self.is_tits(states)
        if tits_pairs:
            print("TITS pairs (u -> v) where u < v:", tits_pairs)
        else:
            print("No TITS pairs found.")

        not_pairs = self.is_not(states)
        if not_pairs:
            print("NOT pairs (u <-> not v):", not_pairs)
        else:
            print("No NOT pairs found.")

        and_triples = self.is_and(states)
        if and_triples:
            print("AND triples (u AND v = w):", and_triples)
        else:
            print("No AND triples found.")


# Example usage (you can keep this or remove it, the states.py or modern_states.py will use the class)
# if __name__ == "__main__":
#     # hyperedges = [
#     #     [1, 2, 3], [1, 4, 3], [2, 4, 5]
#     # ]
#     # hg = Hypergraph(num_vertices=5, hyperedges=hyperedges)
#     # hg.analyze()
# # Example usage
#     hyperedges = [
#         [1, 2, 3], [1, 4, 5], [4, 6, 7], [8, 9, 31], [7, 8, 23], [17, 9, 2],
#         [7, 33, 10], [28, 17, 18], [5, 18, 19], [19, 20, 31], [19, 21, 23],
#         [4, 14, 15], [15, 20, 26], [14, 17, 35], [13, 15, 16], [16, 22, 36],
#         [21, 22, 27], [3, 13, 23], [2, 22, 24], [11, 12, 13], [10, 11, 34],
#         [24, 25, 32], [5, 11, 25], [9, 12, 29], [6, 24, 30], [2, 10, 20]
#     ]

#     hg = Hypergraph(num_vertices=36, hyperedges=hyperedges)
#     hg.analyze()
