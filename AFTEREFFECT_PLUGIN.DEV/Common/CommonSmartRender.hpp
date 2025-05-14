#ifndef __IMAGE_LAB2_ADOBE_AE_SMART_RENDER_COMMON_APIS__
#define __IMAGE_LAB2_ADOBE_AE_SMART_RENDER_COMMON_APIS__

#include "AE_Effect.h"
#include "FastAriphmetics.hpp"


inline PF_Boolean IsEmptyRect (const PF_LRect* r) noexcept
{
    return (r->left >= r->right) || (r->top >= r->bottom);
}


inline void UnionLRect (const PF_LRect* src, PF_LRect* dst) noexcept
{
    if (IsEmptyRect(dst))
    {
        *dst = *src;
    }
    else if (!IsEmptyRect(src))
    {
        dst->left   = FastCompute::Min(dst->left,   src->left);
        dst->top    = FastCompute::Min(dst->top,    src->top);
        dst->right  = FastCompute::Min(dst->right,  src->right);
        dst->bottom = FastCompute::Min(dst->bottom, src->bottom);
    }
    return;
}


#endif // __IMAGE_LAB2_ADOBE_AE_SMART_RENDER_COMMON_APIS__