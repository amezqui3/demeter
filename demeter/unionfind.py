class UnionFind:
    """
    Define a union-find data structure. Also known as disjoint-set.
    The union-find structure groups n elements into a collection of disjoint sets.
    The basic operations consist of
       FIND  the unique set that contains the given element
       UNION of two sets

    The data structure mantains a collection S={S_i} of disjoint sets where
    each of them has an element that works as the representative.

    This particular implementation can be thought as a disjoint-set forest,
    where each element can be thought as a vertex and each disjoint set
    as a tree.
    """

    def __init__(self,mergeF=None):
        """
        Constructor of the union-find data structure

        Attributes:
            `V` : A dictionary of vertices
            `Vl`: A list containing the dictionary keys
            `parent`: A list of the parent node of the vertex
            `size`: List of numbers of vertices hanging from each vertex
            `mergeF`: (Optional) merge function of the data when merging two trees
            `data`: (Optional) Dictionary with additional information carried by each vertex
        """
        self.V = dict()
        self.Vl = []
        self.parent = []
        self.size = []
        self.mergeF = mergeF
        self.data = dict()

    def add(self,v,data=None):
        """
        Add a new element `v` to the union-find

        Parameters:
            `v`: Vertex to be inserted
            `data`: (Optional) information associated to the vertex
        """
        if v not in self.V:
            i = len(self.V)
            self.V[v] = i
            self.Vl.append(v)
            self.parent.append(i)
            self.size.append(1)
            self.data[v] = data

    def find_parent(self,v):
        """
        Find the root of the tree where v is located.
        Update the parent information for each vertex

        Parameters:
            `v`: Vertex
        """
        i = self.V[v]
        p = i
        while self.parent[p]!=p:
            p = self.parent[p]
        while i!=p:
            i,j = self.parent[i],i
            self.parent[j] = p
        return self.Vl[p]

    def find_size(self,v):
        """
        Returns the number of vertices hanging below v

        Parameters:
            `v`: vertex
        """
        return self.size[self.V[self.find_parent(v)]]

    # returns (new root,merged root)
    def merge(self,u,v):
        """
        The tree containing vertex u is merged INTO the tree containing vertex v.
        """

        su = self.find_size(u)
        sv = self.find_size(v)
        pu = self.parent[self.V[u]]
        pv = self.parent[self.V[v]]
        if pu==pv:
            return (self.Vl[pu],None)
        d = self.mergeF(self.data[self.Vl[pu]],self.data[self.Vl[pv]])
        if sv<=su:
            pu,pv,su,sv = pv,pu,sv,su
        self.parent[pv] = pu
        self.size[pu] = su+sv
        self.data[self.Vl[pu]] = d
        return (self.Vl[pu],self.Vl[pv])

    def getData(self,v):
        return self.data[self.find_parent(v)]

def mergeF(a,b):
    m,e = max((a['max'],a['elder']),(b['max'],b['elder']))
    return {'max':m,'elder':e}

def persistence(f):
    """
    Computes the 0D persistence of a given filtration. Filtration values are
    passed in an array-like structure `f`. These values are later sorted from
    high to low and we compute the persistence of connected components.

    Whenever two connected components merge, we record the filter value difference
    and store the merge point in a list `pairs`.

    Parameters:
        `f`: array-like with filtration values.

    Returns:
        `pairs`: list of tuples (persistence, death, birth). A connected-component
        with infinite persistence is appended at the end.
    """
    fi = sorted(list(zip(f,range(len(f)))),reverse=True)
    uf = UnionFind(mergeF)
    pairs = []
    for v,i in fi:
        uf.add(i,{'max':v,'elder':i})
        if i-1 in uf.V and i+1 in uf.V:
            a = uf.getData(i-1)
            b = uf.getData(i+1)
            d,j = min((a['max'],a['elder']),(b['max'],b['elder']))
            pairs.append((d-v,i,j))
        if i-1 in uf.V:
            uf.merge(i-1,i)
        if i+1 in uf.V:
            uf.merge(i,i+1)
    pairs.append((float('inf'),None,fi[0][1]))
    return pairs

def rel_persistence(f,threshold=1e4):
    """
    Same idea as `persistence`, except that instead of recording persistance
    (death - birth), we record persistence relative to death, that is
    (death - birth)/death.

    We can also limit ourselves to only record those critical points where
    persistence is larger than some set threshold in order to avoid noise such
    as birth=0, death=3, which results in a relative persistence of 1.

    Parameters:
        `f`: array-like with filtration values.
        `threshold`: scalar. Consider only critical points whose persistence is larger
        than a fixed threshold.

    Returns:
        `pairs`: list of tuples (persistence/death, death, birth).
        A connected-component with infinite persistence is appended at the end.
    """

    fi = sorted(list(zip(f,range(len(f)))),reverse=True)
    uf = UnionFind(mergeF)
    pairs = []
    for v,i in fi:
        uf.add(i,{'max':v,'elder':i})
        if i-1 in uf.V and i+1 in uf.V:
            a = uf.getData(i-1)
            b = uf.getData(i+1)
            d,j = min((a['max'],a['elder']),(b['max'],b['elder']))
            if d-v > threshold:
                pairs.append(((d-v)/d,i,j))
        if i-1 in uf.V:
            uf.merge(i-1,i)
        if i+1 in uf.V:
            uf.merge(i,i+1)
    pairs.append((float('inf'),None,fi[0][1]))
    return pairs
