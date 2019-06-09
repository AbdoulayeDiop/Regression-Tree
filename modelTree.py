import numpy as np

from region import Region1, Region2


class Model():
    def __init__(self, minSize, eval, norm, alpha):
        self.minSize = minSize
        self.axis = None
        self.subAxis = None
        self.root = None
        self.eval = eval
        self.norm = norm
        self.alpha = alpha

    def bestSeparation(self, parentRegion):
        opt = np.inf
        jOpt = None
        sOpt = None
        for j in range(len(parentRegion.getPopulation()[0])):
            svalues = np.sort(parentRegion.getPopulation()[:, j])
            maxs = max(svalues)
            n = len(svalues)
            while n > 0 and svalues[-1] == maxs:
                svalues = svalues[:-1]
                n -= 1
            for s in svalues:
                r1 = Region1(self, j, s, elements=parentRegion.getPopulation())
                r2 = Region2(self, j, s, elements=parentRegion.getPopulation())
                loss = r1.getLoss(r1.value) + r2.getLoss(r2.value)
                if loss < opt:
                    opt = loss
                    jOpt = j
                    sOpt = s
        if jOpt != None:
            r1Opt = Region1(self, jOpt, sOpt, elements=parentRegion.getPopulation())
            r2Opt = Region2(self, jOpt, sOpt, elements=parentRegion.getPopulation())
            return r1Opt, r2Opt
        else:
            return None, None

    def critereCout(self, t, firstTerm=False):
        leaves = t.getLeaves()
        s = 0
        n = len(leaves)
        print("n :", n)
        for node in leaves:
            s += node.getLoss(node.value)
        if not firstTerm:
            return s + self.alpha * n
        else:
            return s

    def copy(self, region):
        if type(region) == Region1:
            r = Region1(self, region.j, region.s, elements=region.getPopulation())
        else:
            r = Region2(self, region.j, region.s, elements=region.getPopulation())
        if not region.is_leaf:
            for node in region.children:
                newnode = self.copy(node)
                newnode.parent = r
        return r

    def elager(self, t):
        tOpt = self.copy(t)
        currentTree = self.copy(t)
        cOpt = self.critereCout(t)
        while not currentTree.is_leaf:
            leaves = currentTree.getLeaves()
            firstTerm = np.inf
            bestIndex = 0
            for i in range(0, len(leaves), 2):
                p = leaves[i].parent
                savechildren = [node for node in p.children]
                p.children = []
                fs = self.critereCout(currentTree, firstTerm=True)
                if fs < firstTerm:
                    firstTerm = fs
                    bestIndex = i
                p.children = savechildren
            leaves[bestIndex].parent.children = []
            c = self.critereCout(currentTree)
            print("c :", c)
            print("cOpt :", cOpt)
            if c < cOpt:
                print("--------------------------------------------- c :", c)
                cOpt = c
                tOpt = self.copy(currentTree)
        return tOpt

    def classify(self, data):
        root = Region1(self, 1, np.inf, elements=data)

        def divideRecursivelly(region):
            if region.getSize() > self.minSize:
                print(".................")
                r1, r2 = self.bestSeparation(region)
                if r1:
                    r1.parent = region
                    divideRecursivelly(r1)
                    r2.parent = region
                    divideRecursivelly(r2)

        divideRecursivelly(root)
        root = self.elager(root)
        self.root = root
        return root

    def evaluate(self, elt):
        def recEvaluate(region):
            if not region.children:
                return region
            else:
                r1, r2 = region.children
                if elt[r1.j] <= r1.s:
                    return recEvaluate(r1)
                else:
                    return recEvaluate(r2)

        sol = recEvaluate(self.root)
        norm1 = self.norm(elt - sol.value)
        norm2 = self.norm(sol.value)
        print("region : ", sol, "     acc : ", 1 - norm1 / norm2)
        return sol

    def evaluate2(self, elements):
        solList = []
        loss = 0

        def recEvaluate(elt, region):
            if not region.children:
                return region
            else:
                r1, r2 = region.children
                if elt[r1.j] <= r1.s:
                    return recEvaluate(elt, r1)
                else:
                    return recEvaluate(elt, r2)

        for elt in elements:
            sol = recEvaluate(elt, self.root)
            solList.append(sol.value)
            loss += self.norm(self.eval(elt) - sol.value)
        loss = loss / len(elements)
        print("loss : ", loss)
        return solList, loss
