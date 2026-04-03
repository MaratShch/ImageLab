#pragma once
#include <vector>
#include <cstdint>
#include "Common.hpp" // For A_long

template <typename T>
struct Edge
{
    A_long target;
    T weight;
};

template <typename T>
class SparseMatrix
{
public:
    A_long numNodes;
    std::vector<std::vector<Edge<T>>> adj;

    // Constructor pre-allocates the outer array to the number of pixels
    explicit SparseMatrix(A_long nodes) : numNodes(nodes)
	{
        adj.resize(nodes);
    }

    // Add a connection between two pixels
    inline void add_edge(A_long u, A_long v, T w)
	{
        adj[u].push_back({v, w});
    }
};