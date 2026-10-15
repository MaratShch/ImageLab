// ---------------------------------------------------------------------------
//  AlgoFrequency.cpp
//
//  Implementation of AlgoFrequency.hpp. ONE translation unit for both engines:
//  the lane type and its two per-bin helpers come from AlgoFftLane.hpp, which
//  the AVX2 tree overlays. Raw pointers, explicit geometry, no allocation.
// ---------------------------------------------------------------------------

#include "AlgoFrequency.hpp"
#include "AlgoFftLane.hpp"

#include <cmath>


namespace
{
    using L = AlgoFftLane;
    using V = L::V;
    using Plan = FourierTransform::PlanReal2D<AlgoType>;

    constexpr HighPrecType kTwoPi = 6.283185307179586476925286766559;

    inline std::size_t RoundLine (const std::size_t b) noexcept
    {
        return (b + static_cast<std::size_t>(CACHE_LINE) - 1u) & ~(static_cast<std::size_t>(CACHE_LINE) - 1u);
    }

    // ----------------------------------------------------------------------
    //  The transfer, evaluated K bins at a time inside the inverse transform's
    //  column gather. Weights are folded into the gy tables at setup.
    // ----------------------------------------------------------------------
    struct TransferEval
    {
        const AlgoFreqState* st;
        int32_t  nA;
        int32_t  nB;
        bool     law;
        AlgoType halfQ;
        AlgoType invF50Sq;
        bool     shift;

        inline V Mul (const int32_t r, const int32_t c0, const V& x) const noexcept
        {
            const int32_t SP2 = 2 * st->specPitch;
            const int32_t H   = st->sizeY;
            V y = x;
            if (nA > 0)
            {
                V acc = L::MulR(L::Load(st->gxA + 2 * c0), st->gyA[r]);
                for (int32_t i = 1; i < nA; i++)
                    acc = L::Add(acc, L::MulR(L::Load(st->gxA + i * SP2 + 2 * c0), st->gyA[i * H + r]));
                y = L::MulRV(y, acc);
            }
            if (nB > 0)
            {
                V acc = L::MulR(L::Load(st->gxB + 2 * c0), st->gyB[r]);
                for (int32_t i = 1; i < nB; i++)
                    acc = L::Add(acc, L::MulR(L::Load(st->gxB + i * SP2 + 2 * c0), st->gyB[i * H + r]));
                y = L::MulRV(y, acc);
            }
            if (law)
            {
                const V f2 = AlgoFftLaneMath::AddR(L::Load(st->fx2mmD + 2 * c0), st->fy2mm[r]);
                y = L::MulRV(y, AlgoFftLaneMath::LawPairs(f2, halfQ, invF50Sq));
            }
            if (shift)
            {
                const V w = L::CMul(L::Load(st->shX + 2 * c0), st->shY[2 * r], st->shY[2 * r + 1]);
                y = L::CMulV(y, w);
            }
            return y;
        }
    };

    // Separable factors below this are stored as exactly zero. A wide Gaussian
    // (halation, flare) drives exp(-a f^2) far below 1e-38 within a few bins,
    // and float32 then carries DENORMALS through every product of the column
    // gather -- measured 33.8 ms against 13.6 ms for one 1920x1080 AVX2 filter
    // with a four-lobe halation kernel (bench machine, not the i7-7700K). 1e-18 keeps every product of two factors (>= 1e-36)
    // normal in float; what it discards is below 1e-18 of the DC response, far
    // under float32 resolution of the transfer itself. Applied identically in
    // both engines so the two keep one flow.
    constexpr HighPrecType kSeparableFloor = 1.0e-18;

    inline AlgoType Floored (const HighPrecType v) noexcept
    {
        return (v < kSeparableFloor && v > -kSeparableFloor) ? static_cast<AlgoType>(0) : static_cast<AlgoType>(v);
    }

    // sum_i w_i exp(-a_i fy^2) exp(-a_i fx^2), weights folded into gy.
    void BuildSum (const AlgoFreqState& st, const AlgoFreqGaussSum& s, AlgoType* gy, AlgoType* gx) noexcept
    {
        const int32_t H   = st.sizeY;
        const int32_t SP  = st.specPitch;
        const int32_t SW  = st.sizeX / 2 + 1;
        for (int32_t i = 0; i < s.n; i++)
        {
            AlgoType* RESTRICT y = gy + static_cast<std::ptrdiff_t>(i) * H;
            AlgoType* RESTRICT x = gx + static_cast<std::ptrdiff_t>(i) * 2 * SP;
            const HighPrecType a = s.a[i];
            const HighPrecType w = s.w[i];
            for (int32_t r = 0; r < H; r++)
                y[r] = Floored(w * std::exp(-a * static_cast<HighPrecType>(st.fy2mm[r])));
            for (int32_t c = 0; c < SW; c++)
            {
                const AlgoType v = Floored(std::exp(-a * static_cast<HighPrecType>(st.fx2mmD[2 * c])));
                x[2 * c] = v;
                x[2 * c + 1] = v;
            }
            for (int32_t c = SW; c < SP; c++) { x[2 * c] = static_cast<AlgoType>(0); x[2 * c + 1] = static_cast<AlgoType>(0); }
        }
    }
}


// ---------------------------------------------------------------------------
//  Sizing
// ---------------------------------------------------------------------------
std::size_t AlgoFreqPlanBytes (const int32_t sizeX, const int32_t sizeY) noexcept
{
    return RoundLine(FourierTransform::PlanReal2DBuild<AlgoType>(nullptr, sizeX, sizeY, nullptr));
}

std::size_t AlgoFreqSpecBytes (const int32_t sizeX, const int32_t sizeY) noexcept
{
    return static_cast<std::size_t>(ALGO_FREQ_SPECTRA)
         * RoundLine(FourierTransform::Real2DSpecElements<AlgoType>(sizeX, sizeY) * sizeof(AlgoType));
}

std::size_t AlgoFreqWorkBytes (const int32_t sizeX, const int32_t sizeY) noexcept
{
    // The real 2D plan needs max(W, H) lane vectors plus its Bluestein work,
    // 2 m with m = next power of two >= 2 n - 1, i.e. below 8 n. Sized at the
    // widest lane vector so one arena serves either engine.
    const int32_t nmax = (sizeX > sizeY) ? sizeX : sizeY;
    const std::size_t vecs = 9u * static_cast<std::size_t>(nmax);
    return RoundLine(vecs * ALGO_FREQ_LANE_BYTES + FourierTransform::kPlanAlign);
}

std::size_t AlgoFreqTableBytes (const int32_t sizeX, const int32_t sizeY) noexcept
{
    const std::size_t H  = static_cast<std::size_t>(sizeY);
    const std::size_t SP = static_cast<std::size_t>(FourierTransform::Real2DSpecPitch<AlgoType>(sizeX));
    const std::size_t T  = static_cast<std::size_t>(ALGO_FREQ_MAX_TERMS);
    const std::size_t elems = H + SP + H + 2 * SP           // fy, fx, fy2mm, fx2mmD
                            + 2 * (T * H + T * 2 * SP)      // gyA gxA gyB gxB
                            + 2 * H + 2 * SP                // shY shX
                            + 16 * 8;                       // per-table alignment slack
    return RoundLine(elems * sizeof(AlgoType));
}


// ---------------------------------------------------------------------------
//  Construction
// ---------------------------------------------------------------------------
bool AlgoFreqInit (AlgoFreqState& st, const int32_t sizeX, const int32_t sizeY,
                   void* planMem, void* specMem, void* workMem, void* tableMem) noexcept
{
    if (sizeX < 1 || sizeY < 1 || nullptr == planMem || nullptr == specMem
        || nullptr == workMem || nullptr == tableMem)
        return false;

    FourierTransform::PlanReal2DBuild<AlgoType>(&st.plan, sizeX, sizeY, planMem);
    st.sizeX = sizeX;
    st.sizeY = sizeY;
    st.specPitch = st.plan.specPitch;
    {
        const std::size_t one = RoundLine(FourierTransform::Real2DSpecElements<AlgoType>(sizeX, sizeY) * sizeof(AlgoType));
        for (int32_t i = 0; i < ALGO_FREQ_SPECTRA; i++)
            st.spec[i] = reinterpret_cast<AlgoType*>(static_cast<unsigned char*>(specMem) + static_cast<std::size_t>(i) * one);
    }
    st.work = workMem;

    const int32_t H  = sizeY;
    const int32_t SP = st.specPitch;
    const int32_t T  = ALGO_FREQ_MAX_TERMS;
    AlgoType* p = static_cast<AlgoType*>(tableMem);
    auto take = [&p](const int32_t n) noexcept -> AlgoType*
    {
        AlgoType* q = p;
        // keep every table on an 8-element boundary (32 bytes in float, 64 in double)
        p += (static_cast<std::ptrdiff_t>(n) + 7) & ~static_cast<std::ptrdiff_t>(7);
        return q;
    };
    st.fy     = take(H);
    st.fx     = take(SP);
    st.fy2mm  = take(H);
    st.fx2mmD = take(2 * SP);
    st.gyA    = take(T * H);
    st.gxA    = take(T * 2 * SP);
    st.gyB    = take(T * H);
    st.gxB    = take(T * 2 * SP);
    st.shY    = take(2 * H);
    st.shX    = take(2 * SP);

    // numpy.fft.fftfreq(H) / rfftfreq(W), cycles per pixel
    for (int32_t r = 0; r < H; r++)
    {
        const int32_t k = (r <= (H - 1) / 2) ? r : (r - H);
        st.fy[r] = static_cast<AlgoType>(static_cast<HighPrecType>(k) / static_cast<HighPrecType>(H));
    }
    const int32_t SW = sizeX / 2 + 1;
    for (int32_t c = 0; c < SP; c++)
        st.fx[c] = (c < SW) ? static_cast<AlgoType>(static_cast<HighPrecType>(c) / static_cast<HighPrecType>(sizeX))
                            : static_cast<AlgoType>(0);
    return true;
}


// ---------------------------------------------------------------------------
//  Per frame
// ---------------------------------------------------------------------------
void AlgoFreqBeginFrame (const AlgoFreqState& st, const AlgoType pxPerMm) noexcept
{
    // film_sim: fy_mm = fy * px_per_mm (float32), f_mm^2 = fy_mm^2 + fx_mm^2
    for (int32_t r = 0; r < st.sizeY; r++)
    {
        const AlgoType v = st.fy[r] * pxPerMm;
        st.fy2mm[r] = v * v;
    }
    for (int32_t c = 0; c < st.specPitch; c++)
    {
        const AlgoType v = st.fx[c] * pxPerMm;
        st.fx2mmD[2 * c] = v * v;
        st.fx2mmD[2 * c + 1] = v * v;
    }
}


// ---------------------------------------------------------------------------
//  Filter
// ---------------------------------------------------------------------------
namespace
{
    TransferEval PrepareTransfer (const AlgoFreqState& st, const AlgoFreqTransfer& t) noexcept
    {
        TransferEval ev;
        ev.st = &st;
        ev.nA = t.sumA.n;
        ev.nB = t.sumB.n;
        ev.law = (0 != t.hasLaw);
        ev.halfQ = static_cast<AlgoType>(0.5 * t.lawQ);
        ev.invF50Sq = (t.lawF50 > 0.0) ? static_cast<AlgoType>(1.0 / (t.lawF50 * t.lawF50)) : static_cast<AlgoType>(0);
        ev.shift = (0 != t.hasShift);

        if (ev.nA > 0) BuildSum(st, t.sumA, st.gyA, st.gxA);
        if (ev.nB > 0) BuildSum(st, t.sumB, st.gyB, st.gxB);
        if (ev.shift)
        {
            // exp(-2 pi j fy dy) per row, exp(-2 pi j fx dx) per column
            for (int32_t r = 0; r < st.sizeY; r++)
            {
                const HighPrecType ph = -kTwoPi * static_cast<HighPrecType>(st.fy[r]) * t.shiftDy;
                st.shY[2 * r]     = static_cast<AlgoType>(std::cos(ph));
                st.shY[2 * r + 1] = static_cast<AlgoType>(std::sin(ph));
            }
            for (int32_t c = 0; c < st.specPitch; c++)
            {
                const HighPrecType ph = -kTwoPi * static_cast<HighPrecType>(st.fx[c]) * t.shiftDx;
                st.shX[2 * c]     = static_cast<AlgoType>(std::cos(ph));
                st.shX[2 * c + 1] = static_cast<AlgoType>(std::sin(ph));
            }
        }
        return ev;
    }

    inline AlgoType InverseScale (const AlgoFreqState& st) noexcept
    {
        return static_cast<AlgoType>(1.0 / (static_cast<HighPrecType>(st.sizeX)
                                          * static_cast<HighPrecType>(st.sizeY)));
    }
}


void AlgoFreqFilterPlane (const AlgoFreqState& st, const AlgoType* src, AlgoType* dst,
                          const int32_t pitch, const AlgoFreqTransfer& t) noexcept
{
    const TransferEval ev = PrepareTransfer(st, t);
    FourierTransform::Real2DForward<L>(st.plan, src, pitch, st.spec[0], st.work);
    FourierTransform::Real2DInverse<L>(st.plan, st.spec[0], ev, st.spec[0], dst, pitch, InverseScale(st), st.work);
}


void AlgoFreqForward (const AlgoFreqState& st, const AlgoType* src, const int32_t pitch,
                      const int32_t slot) noexcept
{
    FourierTransform::Real2DForward<L>(st.plan, src, pitch, st.spec[slot], st.work);
}


void AlgoFreqInverse (const AlgoFreqState& st, const int32_t slot, const AlgoFreqTransfer& t,
                      AlgoType* dst, const int32_t pitch) noexcept
{
    const TransferEval ev = PrepareTransfer(st, t);
    FourierTransform::Real2DInverse<L>(st.plan, st.spec[slot], ev, st.spec[slot], dst, pitch, InverseScale(st), st.work);
}


void AlgoFreqMeanMix3 (const AlgoFreqState& st, const HighPrecType keep, const HighPrecType mix,
                       const AlgoFreqTransfer& t) noexcept
{
    const TransferEval ev = PrepareTransfer(st, t);
    const AlgoType k = static_cast<AlgoType>(keep);
    const AlgoType m = static_cast<AlgoType>(mix / 3.0);
    constexpr int32_t K = L::kLanes;
    const std::ptrdiff_t SP2 = 2 * static_cast<std::ptrdiff_t>(st.specPitch);
    const int32_t SW = st.sizeX / 2 + 1;
    for (int32_t r = 0; r < st.sizeY; r++)
    {
        AlgoType* p0 = st.spec[0] + r * SP2;
        AlgoType* p1 = st.spec[1] + r * SP2;
        AlgoType* p2 = st.spec[2] + r * SP2;
        for (int32_t c0 = 0; c0 < SW; c0 += K)
        {
            const V d0 = L::Load(p0 + 2 * c0);
            const V d1 = L::Load(p1 + 2 * c0);
            const V d2 = L::Load(p2 + 2 * c0);
            // m * T * (D0 + D1 + D2), shared by the three records
            const V shared = L::MulR(ev.Mul(r, c0, L::Add(L::Add(d0, d1), d2)), m);
            L::Store(p0 + 2 * c0, L::Add(L::MulR(d0, k), shared));
            L::Store(p1 + 2 * c0, L::Add(L::MulR(d1, k), shared));
            L::Store(p2 + 2 * c0, L::Add(L::MulR(d2, k), shared));
        }
    }
}
