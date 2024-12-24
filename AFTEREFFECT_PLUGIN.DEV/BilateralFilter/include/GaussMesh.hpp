#ifndef __IMAGE_LAB_BILATERAL_FILTER_GAUSS_MESH__
#define __IMAGE_LAB_BILATERAL_FILTER_GAUSS_MESH__

#include <atomic>
#include <mutex>
#include <array>
#include "Common.hpp"
#include "ClassRestrictions.hpp"
#include "BilateralFilterEnum.hpp"

class GaussMesh;

GaussMesh* getMeshHandler(void);
GaussMesh* CreateGaussMeshHandler(void);
void  ReleaseGaussMeshHandler(void* p);

using MeshT = float;

class GaussMesh final
{
public:
    static GaussMesh* getGaussMeshInstance(void)
    {
        GaussMesh* iGaussMesh = s_instance.load(std::memory_order_acquire);
        if (nullptr == iGaussMesh)
        {
            std::lock_guard<std::mutex> myLock(s_protectMutex);
            iGaussMesh = s_instance.load(std::memory_order_relaxed);
            if (nullptr == iGaussMesh)
            {
                iGaussMesh = new GaussMesh();
                iGaussMesh->InitMesh();
                s_instance.store(iGaussMesh, std::memory_order_release);
            }
        }
        return iGaussMesh;
    } /* static iGaussMesh* getInstance() */

    const MeshT* geCenterMesh(void) const noexcept
    {
        constexpr size_t meshCenter = static_cast<size_t>(maxWindowSize * bilateralMaxRadius + bilateralMaxRadius + 1);
        return &m_Mesh[meshCenter];
    }

    const MeshT* getMesh (const A_long& radius, A_long& meshPitch) const noexcept
    {
        constexpr size_t meshCenter = static_cast<size_t>(maxWindowSize * bilateralMaxRadius + bilateralMaxRadius + 1);
        const size_t meshStart = meshCenter - radius - (maxWindowSize * radius) - 1;
        meshPitch = maxWindowSize;
        return &m_Mesh[meshStart];
     }

    const A_long getMeshPitch(void) const noexcept {return maxWindowSize;}

    

private:
    GaussMesh(void) {};
    ~GaussMesh(void){};

    CLASS_NON_COPYABLE(GaussMesh);
    CLASS_NON_MOVABLE (GaussMesh);

    void InitMesh(void);

    static std::atomic<GaussMesh*> s_instance;
    static std::mutex s_protectMutex;

    std::array<MeshT, maxMeshSize>m_Mesh;

}; // class GaussMesh


#endif // __IMAGE_LAB_BILATERAL_FILTER_GAUSS_MESH__