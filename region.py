import numpy as np
from anytree.node import node


class Region1(node.Node):
    def __init__(self, model, j, s, parent=None, elements=[], getlabel=None):
        if getlabel:
            name = getlabel(j) + "\n <= " + str(s)
        else:
            name = str(j) + "\n <= " + str(s)
        super().__init__(name, parent=parent) if parent else super().__init__(name)
        self.model = model
        self.j = j
        self.s = s
        if parent:
            self.population = np.array([elt for elt in parent.getPopulation() if elt[j] <= s])
        else:
            self.population = np.array([elt for elt in elements if elt[j] <= s])

        self.name += "\n nb : " + str(self.getSize())
        self.value = self.getMean()
        self.name += "\n" + str(self.value)

    def getPopulation(self):
        return self.population

    def getSize(self):
        return len(self.population)

    def getMean(self):
        return sum(
            [self.model.eval(elt) for elt in self.getPopulation()]) / self.getSize() if self.getSize() != 0 else 0

    def getElementSize(self):
        return len(self.getPopulation()[0])

    def getLoss(self, val):
        return sum([self.model.norm(self.model.eval(elt) - val) for elt in self.getPopulation()])

    def getLeaves(self):
        return np.array(self.leaves)

    def __repr__(self):
        return self.name


class Region2(node.Node):
    def __init__(self, model, j, s, parent=None, elements=[], getlabel=None):
        if getlabel:
            name = getlabel(j) + "\n > " + str(s)
        else:
            name = str(j) + "\n > " + str(s)
        super().__init__(name, parent=parent) if parent else super().__init__(name)
        self.model = model
        self.j = j
        self.s = s
        if parent:
            self.population = np.array([elt for elt in parent.getPopulation() if elt[j] > s])
        else:
            self.population = np.array([elt for elt in elements if elt[j] > s])

        self.name += "\n nb : " + str(self.getSize())
        self.value = self.getMean()
        self.name += "\n" + str(self.value)

    def getPopulation(self):
        return self.population

    def getSize(self):
        return len(self.population)

    def getSize(self):
        return len(self.population)

    def getMean(self):
        return sum(
            [self.model.eval(elt) for elt in self.getPopulation()]) / self.getSize() if self.getSize() != 0 else 0

    def getElementSize(self):
        return len(self.getPopulation()[0])

    def getLoss(self, val):
        return sum([self.model.norm(self.model.eval(elt) - val) for elt in self.getPopulation()])

    def getLeaves(self):
        return np.array(self.leaves)

    def __repr__(self):
        return self.name
