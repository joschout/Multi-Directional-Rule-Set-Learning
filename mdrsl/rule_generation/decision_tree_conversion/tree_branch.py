from typing import List, Optional
from .tree_edge import TreeEdge


class TreeBranch:
    def __init__(self, parent_tree_branch: Optional['TreeBranch'], edge: TreeEdge):
        self.parent_tree_branch: Optional['TreeBranch'] = parent_tree_branch
        self.new_literal: TreeEdge = edge

    def to_list(self) -> List[TreeEdge]:
        return self._to_list_recursive()

    def _to_list_recursive(self) -> List[TreeEdge]:
        if self.parent_tree_branch is None:
            return [self.new_literal]
        else:
            previous_list = self.parent_tree_branch._to_list_recursive()
            previous_list.append(self.new_literal)
            return previous_list
