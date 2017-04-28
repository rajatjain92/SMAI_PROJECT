def kmeans(data, k = 2, dist_func = None, max_iter = 10):
    ##TODO: set optimal number of iterations and stop when results are satisfied
    "Computes k-means for the given data"

    def dist(vec1, vec2):
        "distance function used by k-means"
        return np.linarg.norm(vec2 - vec1)

    def get_closest(vec):
        dst = [dist_func(vec, k_means[:, i]) for i in range(k)]
        return np.argmin(dst)
    
    if dist_func is None:
        dist_func = dist

    d, n = data.shape
    init_random = np.random.choice(np.arange(n), k)
    k_means = data[:, init_random]

    for itr in range(max_iter):
        assigns = [[] for i in range(k)]
        
        # find new assignments
        for vec_no in range(n):
            vec = data[:, vec_no]
            assign[get_closest(vec)].append(vec_no)

        # find new means
        for i in range(k):
            ##TODO: when assign is zero, we need to re-initialize it randomly
            k_means[:, i] = np.mean(data[:, assign], 1)

    # construct the assignment vector
    assignments = [0] * n
    for i in range(k):
        l = assigns[i]
        for x in l:
            assignments[x] = i

    return (k_means, assignments)
