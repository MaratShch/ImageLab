#define FAST_COMPUTE_EXTRA_PRECISION

#include "FastAriphmetics.hpp"
#include "GaussMesh.hpp"

GaussMesh* CreateGaussMeshHandler (void)
{
    return GaussMesh::getGaussMeshInstance();
}

void  ReleaseGaussMeshHandler (void* p)
{
    /* nothing to do */
    (void)p;
    return;
}

void GaussMesh::InitMesh(void)
{
    constexpr MeshT sigma{ 3 };
    constexpr MeshT divider = sigma * sigma * static_cast<MeshT>(2);
    A_long k = 0;

    for (A_long j = -bilateralMaxRadius; j <= bilateralMaxRadius; j++)
    {
        __VECTOR_ALIGNED__
        for (A_long i = -bilateralMaxRadius; i <= bilateralMaxRadius; i++)
        {
            const MeshT meshIdx = static_cast<MeshT>((i * i) + (j * j));
            m_Mesh[k] = FastCompute::Exp(-meshIdx / divider);
            k++;
        }
    }
    return;
}


std::atomic<GaussMesh*> GaussMesh::s_instance;
std::mutex GaussMesh::s_protectMutex;

