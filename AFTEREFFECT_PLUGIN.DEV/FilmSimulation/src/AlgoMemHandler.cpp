// ---------------------------------------------------------------------------
//  AlgoMemHandler.cpp
//
//  Arena allocation for the film simulation engine.
//
//  ONE block from the pool per frame geometry, sliced into named pointers by a
//  running offset stack. The engine reads those pointers and never allocates,
//  resizes or frees anything.
// ---------------------------------------------------------------------------

#include "AlgoMemHandler.hpp"
#include "ImageLabMemInterface.hpp"

#include <cstdio>    // std::printf, diagnostic path only
#include <vector>


namespace
{
    // ----------------------------------------------------------------------
    //  Largest frame dimension accepted.
    //
    //  65535 keeps every byte count below the range of the int32_t the pool
    //  interface takes, with room to spare, and no real render approaches it. A
    //  larger request is refused rather than silently wrapping an offset.
    // ----------------------------------------------------------------------
    constexpr int32_t ALGO_MAX_DIMENSION = 65535;

    // ----------------------------------------------------------------------
    //  Number of retained stage triples in the pipeline.
    //
    //  Counted here so the reported total can be cross-checked against the number
    //  of offsets actually pushed. A mismatch means a buffer was added to the
    //  struct and forgotten in the offset stack, which would otherwise appear as
    //  two stages sharing memory and quietly corrupting each other.
    // ----------------------------------------------------------------------
    constexpr int32_t ALGO_STAGE_TRIPLE_COUNT = 25;

    // Number of single-plane scratch buffers, and of retained log-exposure and
    // grain planes. Kept as named counts for the same cross-check reason.
    constexpr int32_t ALGO_SCRATCH_SINGLE_COUNT = 7;   // BlurA..DbarBlur
    constexpr int32_t ALGO_SCRATCH_TRIPLE_COUNT = 2;   // LogE, Grain


    // ----------------------------------------------------------------------
    //  The three caller-owned lookup tables.
    //
    //  Built ONCE, on first use, and never modified. The allocator attaches them
    //  to every handler it returns.
    //
    //  Why here rather than left to the caller: an earlier revision made the
    //  caller assign them after this function returned. They defaulted to null,
    //  the engine is forbidden by its own rules from validating them, and missing
    //  the step produced a null dereference on the first frame with no diagnostic
    //  whatsoever. Anything the engine must have and cannot check belongs to the
    //  allocator.
    //
    //  Why a function-local static is legitimate here, given the engine's
    //  no-mutable-state rule:
    //
    //    - This is ALLOCATOR code, not Algorithm_Main. The prohibition on
    //      allocation applies to the per-frame path; this runs once per geometry.
    //    - C++11 onward guarantees function-local static initialisation happens
    //      exactly once and is thread safe, so two host threads reaching the first
    //      render together cannot race or observe a half-built table.
    //    - The tables are const after initialisation. Immutable shared state is
    //      what the reentrancy rule permits; nothing ever writes to them.
    //    - Their lifetime runs to program exit, so the pointers stay valid for as
    //      long as any handler does.
    // ----------------------------------------------------------------------
    const std::vector<film::FilmProfile>& profileTable (void) noexcept
    {
        static const std::vector<film::FilmProfile> table = film::GetFilmDatabase();
        return table;
    }

    const std::vector<film::PrintStock>& printTable (void) noexcept
    {
        static const std::vector<film::PrintStock> table = film::GetPrintStocks();
        return table;
    }

    const std::vector<film::FilmFormat>& formatTable (void) noexcept
    {
        static const std::vector<film::FilmFormat> table = film::GetFilmFormats();
        return table;
    }


    // ----------------------------------------------------------------------
    //  Offset stack.
    //
    //  Hands out byte offsets into the block that does not exist yet. Every slice
    //  is rounded up to a cache line, so two stages writing adjacent buffers
    //  cannot false-share a line, and every plane start is therefore also
    //  32-byte aligned.
    //
    //  A plain struct with one mutable field, local to this translation unit and
    //  used only during allocation. It is not state the engine can see.
    // ----------------------------------------------------------------------
    struct OffsetStack
    {
        std::size_t cursor;
        int32_t     slices;

        // Reserve one buffer of the given byte count and return its offset.
        std::size_t push (const std::size_t bytes) noexcept
        {
            const std::size_t here = cursor;

            // CreateAlignment is a SINGLE-type template, so both arguments must be
            // the same type -- hence the explicit cast on CACHE_LINE, which is an
            // int macro. Note its contract: it returns the quantum when the value is
            // not positive, which is why the sum is formed first and is always > 0.
            cursor = CreateAlignment<std::size_t>(cursor + bytes,
                                                 static_cast<std::size_t>(CACHE_LINE));
            slices++;

            return here;
        }
    };
}


// ---------------------------------------------------------------------------
//  alloc_memory_buffers
// ---------------------------------------------------------------------------
MemHandler alloc_memory_buffers (const int32_t sizeX, const int32_t sizeY)
{
    // Zero-initialised, so every failure path below can simply return it: a
    // caller testing SuperBufferHead sees null and totalSize sees zero.
    MemHandler algoMemHandler{};

    if (sizeX <= 0 || sizeY <= 0 ||
        sizeX > ALGO_MAX_DIMENSION || sizeY > ALGO_MAX_DIMENSION)
        return algoMemHandler;

    // ==================================================================
    // 1. PADDED GEOMETRY
    // ==================================================================
    //
    // Width is padded to a multiple of ALGO_PAD_ELEMS, the LARGER of the two
    // element quanta, so ONE padded width serves both element types: padW
    // multiplied by either sizeof is then a multiple of 32 bytes and every row
    // start of every plane lands on a 32-byte boundary.
    //
    // Height is padded to the same quantum. Not required for row alignment, but it
    // keeps a plane's byte count a round multiple and gives the vertical passes a
    // few spare rows to read without a boundary test once the AVX2 path arrives.
    const int32_t padX = CreateAlignment(sizeX, static_cast<int32_t>(ALGO_PAD_ELEMS));
    const int32_t padY = CreateAlignment(sizeY, static_cast<int32_t>(ALGO_PAD_ELEMS));

    const std::size_t frameElems = static_cast<std::size_t>(padX)
                                 * static_cast<std::size_t>(padY);

    const std::size_t imgPlaneBytes  = frameElems * sizeof(ImgType);
    const std::size_t algoPlaneBytes = frameElems * sizeof(AlgoType);

    // ==================================================================
    // 2. OFFSET STACK
    // ==================================================================
    //
    // Laid out in pipeline order, so the arena's address order mirrors the frame's
    // progress. That is not cosmetic: a stage reading its predecessor's output then
    // walks forward through memory, which is the direction the hardware prefetcher
    // handles best.
    OffsetStack stack{ 0u, 0 };

    // --- boundary images, STORAGE type ---
    const std::size_t offSrcR = stack.push(imgPlaneBytes);
    const std::size_t offSrcG = stack.push(imgPlaneBytes);
    const std::size_t offSrcB = stack.push(imgPlaneBytes);
    const std::size_t offDstR = stack.push(imgPlaneBytes);
    const std::size_t offDstG = stack.push(imgPlaneBytes);
    const std::size_t offDstB = stack.push(imgPlaneBytes);

    // --- the stage triples ---
    // One array of offsets rather than 75 named variables: the mapping below reads
    // them back in the same order, so a triple cannot be reserved and then not
    // mapped, or mapped twice.
    //
    // The array always has 25 entries, whichever mode is in force. What changes is
    // how many DISTINCT planes those 25 entries point at, and that is the whole of
    // the ping/pong optimisation: the alternation is a decision taken here, in the
    // allocator, and nothing else in the engine can tell the difference.
    std::size_t offStage[ALGO_STAGE_TRIPLE_COUNT][3];

#if ALGO_RETAIN_ALL_STAGES

    // Every stage gets its own triple, so every intermediate survives the run and
    // can be dumped without re-running the chain. That is what made stage-by-stage
    // validation against the reference model possible, and it is why this mode
    // still exists.
    for (int32_t s = 0; s < ALGO_STAGE_TRIPLE_COUNT; s++)
        for (int32_t c = 0; c < 3; c++)
            offStage[s][c] = stack.push(algoPlaneBytes);

#else

    // TWO triples, alternating.
    //
    // This is sound because the pipeline is a STRICT LINEAR CHAIN: every stage reads
    // only its immediate predecessor. Verified across all 25 stage calls - not one
    // reaches back two or more steps. The single cross-stage lifetime in the whole
    // engine is Scr_LogE_*, written at stage 7/8 and read at 8b, and that is a
    // scratch triple allocated separately below.
    //
    // With A/B alternation, stage N reads A and writes B while stage N+1 reads B and
    // writes A, so SOURCE AND DESTINATION ARE NEVER THE SAME PLANE. That matters for
    // two reasons beyond correctness of the chain itself:
    //
    //   - eleven stages read a NEIGHBOURHOOD of their source (halation, both MTF
    //     passes, the DIR coupler, grain, duplication, the weave resample). Those
    //     would break under true in-place operation, which is exactly why in-place
    //     is not what this does.
    //
    //   - every stage signature qualifies its source and destination RESTRICT. If
    //     the two ever aliased, that promise would be false and the compiler would be
    //     entitled to any result it liked.
    //
    // Stage 13 deserves a specific note, because it is the sharpest edge here. It
    // copies its source into its destination and then iterates the duplication
    // generations IN PLACE ON ITS OWN DESTINATION. That is correct today and stays
    // correct under alternation precisely because its destination is not its source -
    // so nobody should later "simplify" stage 13 to a single buffer.
    std::size_t offPair[2][3];

    for (int32_t k = 0; k < 2; k++)
        for (int32_t c = 0; c < 3; c++)
            offPair[k][c] = stack.push(algoPlaneBytes);

    // The alternation itself. Odd stages take one triple, even stages the other.
    for (int32_t s = 0; s < ALGO_STAGE_TRIPLE_COUNT; s++)
        for (int32_t c = 0; c < 3; c++)
            offStage[s][c] = offPair[s & 1][c];

#endif

    // --- single-plane scratch ---
    const std::size_t offBlurA    = stack.push(algoPlaneBytes);
    const std::size_t offBlurB    = stack.push(algoPlaneBytes);
    const std::size_t offLuma     = stack.push(algoPlaneBytes);
    const std::size_t offField    = stack.push(algoPlaneBytes);
    const std::size_t offFieldLo  = stack.push(algoPlaneBytes);
    const std::size_t offDbar     = stack.push(algoPlaneBytes);
    const std::size_t offDbarBlur = stack.push(algoPlaneBytes);

    // --- retained scratch triples: log exposure and the grain fields ---
    std::size_t offLogE[3];
    std::size_t offGrain[3];

    for (int32_t c = 0; c < 3; c++) offLogE[c]  = stack.push(algoPlaneBytes);
    for (int32_t c = 0; c < 3; c++) offGrain[c] = stack.push(algoPlaneBytes);

    // Cross-check: every PHYSICAL plane must have been pushed exactly once.
    //
    // The stage term is the physical triple count, not the 25 logical stages, so this
    // check remains meaningful in both modes rather than silently passing in one of
    // them. A mismatch means the push sequence above and the mapping below have
    // drifted apart, which is the one error in this file that would corrupt memory
    // rather than merely waste it.
    constexpr int32_t expectedSlices =
        6 + (ALGO_PHYSICAL_STAGE_TRIPLES * 3)
          + ALGO_SCRATCH_SINGLE_COUNT
          + (ALGO_SCRATCH_TRIPLE_COUNT * 3);

    if (stack.slices != expectedSlices)
    {
        return algoMemHandler;
    }

    // ==================================================================
    // 3. ONE ALLOCATION
    // ==================================================================
    //
    // A whole cache line of slack past the end. The vertical blur passes may read a
    // few elements beyond the active area of the last row - padH exists for exactly
    // that - and the slack guarantees such a read stays inside the block even at
    // the very end of the arena.
    const std::size_t requiredBytes = stack.cursor + CACHE_LINE;

    // The pool interface takes int32_t. A request that cannot be expressed in it is
    // refused here rather than truncated into a much smaller allocation, which
    // would be a heap overflow rather than an out-of-memory failure.
    if (requiredBytes > static_cast<std::size_t>(0x7FFFFFFF))
    {
        return algoMemHandler;
    }

    void* pBlock = nullptr;

    const int32_t blockId = GetMemoryBlock(static_cast<int32_t>(requiredBytes),
                                           static_cast<int32_t>(ALGO_ALIGN_BYTES),
                                           &pBlock);
    if (blockId < 0 || nullptr == pBlock)
        return algoMemHandler;

    // ==================================================================
    // 4. MAP EVERY POINTER
    // ==================================================================
    algoMemHandler.memBlockId      = static_cast<int64_t>(blockId);
    algoMemHandler.SuperBufferHead = static_cast<uint8_t*>(pBlock);
    algoMemHandler.totalSize       = requiredBytes;

    algoMemHandler.padW    = padX;
    algoMemHandler.padH    = padY;
    algoMemHandler.activeW = sizeX;
    algoMemHandler.activeH = sizeY;

    // Attach the lookup tables. Done here rather than left to the caller so the
    // engine can never be handed null tables; see the note on profileTable above.
    algoMemHandler.pProfileDb   = profileTable().data();
    algoMemHandler.profileCount = static_cast<int32_t>(profileTable().size());
    algoMemHandler.pFormatDb    = formatTable().data();
    algoMemHandler.formatCount  = static_cast<int32_t>(formatTable().size());
    algoMemHandler.pPrintDb     = printTable().data();
    algoMemHandler.printCount   = static_cast<int32_t>(printTable().size());

    // Local helpers so each mapping line states only which buffer it is, not how
    // an address is formed.
    auto asImg = [pBlock](const std::size_t off) noexcept -> ImgType*
    {
        return static_cast<ImgType*>(ComputeAddress(pBlock, off));
    };

    auto asAlgo = [pBlock](const std::size_t off) noexcept -> AlgoType*
    {
        return static_cast<AlgoType*>(ComputeAddress(pBlock, off));
    };

    algoMemHandler.Src_R = asImg(offSrcR);
    algoMemHandler.Src_G = asImg(offSrcG);
    algoMemHandler.Src_B = asImg(offSrcB);
    algoMemHandler.Dst_R = asImg(offDstR);
    algoMemHandler.Dst_G = asImg(offDstG);
    algoMemHandler.Dst_B = asImg(offDstB);

    // The 75 stage pointers, in the same order the offsets were pushed. Written out
    // rather than looped, because the struct's fields are named individually and
    // there is no array to iterate; the index literals make a transposition visible.
    algoMemHandler.S02_R  = asAlgo(offStage[ 0][0]);  algoMemHandler.S02_G  = asAlgo(offStage[ 0][1]);  algoMemHandler.S02_B  = asAlgo(offStage[ 0][2]);
    algoMemHandler.S02b_R = asAlgo(offStage[ 1][0]);  algoMemHandler.S02b_G = asAlgo(offStage[ 1][1]);  algoMemHandler.S02b_B = asAlgo(offStage[ 1][2]);
    algoMemHandler.S03_R  = asAlgo(offStage[ 2][0]);  algoMemHandler.S03_G  = asAlgo(offStage[ 2][1]);  algoMemHandler.S03_B  = asAlgo(offStage[ 2][2]);
    algoMemHandler.S03b_R = asAlgo(offStage[ 3][0]);  algoMemHandler.S03b_G = asAlgo(offStage[ 3][1]);  algoMemHandler.S03b_B = asAlgo(offStage[ 3][2]);
    algoMemHandler.S03c_R = asAlgo(offStage[ 4][0]);  algoMemHandler.S03c_G = asAlgo(offStage[ 4][1]);  algoMemHandler.S03c_B = asAlgo(offStage[ 4][2]);
    algoMemHandler.S04_R  = asAlgo(offStage[ 5][0]);  algoMemHandler.S04_G  = asAlgo(offStage[ 5][1]);  algoMemHandler.S04_B  = asAlgo(offStage[ 5][2]);
    algoMemHandler.S05_R  = asAlgo(offStage[ 6][0]);  algoMemHandler.S05_G  = asAlgo(offStage[ 6][1]);  algoMemHandler.S05_B  = asAlgo(offStage[ 6][2]);
    algoMemHandler.S06_R  = asAlgo(offStage[ 7][0]);  algoMemHandler.S06_G  = asAlgo(offStage[ 7][1]);  algoMemHandler.S06_B  = asAlgo(offStage[ 7][2]);
    algoMemHandler.S06b_R = asAlgo(offStage[ 8][0]);  algoMemHandler.S06b_G = asAlgo(offStage[ 8][1]);  algoMemHandler.S06b_B = asAlgo(offStage[ 8][2]);
    algoMemHandler.S07_R  = asAlgo(offStage[ 9][0]);  algoMemHandler.S07_G  = asAlgo(offStage[ 9][1]);  algoMemHandler.S07_B  = asAlgo(offStage[ 9][2]);
    algoMemHandler.S08_R  = asAlgo(offStage[10][0]);  algoMemHandler.S08_G  = asAlgo(offStage[10][1]);  algoMemHandler.S08_B  = asAlgo(offStage[10][2]);
    algoMemHandler.S08b_R = asAlgo(offStage[11][0]);  algoMemHandler.S08b_G = asAlgo(offStage[11][1]);  algoMemHandler.S08b_B = asAlgo(offStage[11][2]);
    algoMemHandler.S09_R  = asAlgo(offStage[12][0]);  algoMemHandler.S09_G  = asAlgo(offStage[12][1]);  algoMemHandler.S09_B  = asAlgo(offStage[12][2]);
    algoMemHandler.S09b_R = asAlgo(offStage[13][0]);  algoMemHandler.S09b_G = asAlgo(offStage[13][1]);  algoMemHandler.S09b_B = asAlgo(offStage[13][2]);
    algoMemHandler.S10_R  = asAlgo(offStage[14][0]);  algoMemHandler.S10_G  = asAlgo(offStage[14][1]);  algoMemHandler.S10_B  = asAlgo(offStage[14][2]);
    algoMemHandler.S10b_R = asAlgo(offStage[15][0]);  algoMemHandler.S10b_G = asAlgo(offStage[15][1]);  algoMemHandler.S10b_B = asAlgo(offStage[15][2]);
    algoMemHandler.S11_R  = asAlgo(offStage[16][0]);  algoMemHandler.S11_G  = asAlgo(offStage[16][1]);  algoMemHandler.S11_B  = asAlgo(offStage[16][2]);
    algoMemHandler.S12_R  = asAlgo(offStage[17][0]);  algoMemHandler.S12_G  = asAlgo(offStage[17][1]);  algoMemHandler.S12_B  = asAlgo(offStage[17][2]);
    algoMemHandler.S13_R  = asAlgo(offStage[18][0]);  algoMemHandler.S13_G  = asAlgo(offStage[18][1]);  algoMemHandler.S13_B  = asAlgo(offStage[18][2]);
    algoMemHandler.S14_R  = asAlgo(offStage[19][0]);  algoMemHandler.S14_G  = asAlgo(offStage[19][1]);  algoMemHandler.S14_B  = asAlgo(offStage[19][2]);
    algoMemHandler.S14b_R = asAlgo(offStage[20][0]);  algoMemHandler.S14b_G = asAlgo(offStage[20][1]);  algoMemHandler.S14b_B = asAlgo(offStage[20][2]);
    algoMemHandler.S14c_R = asAlgo(offStage[21][0]);  algoMemHandler.S14c_G = asAlgo(offStage[21][1]);  algoMemHandler.S14c_B = asAlgo(offStage[21][2]);
    algoMemHandler.S15_R  = asAlgo(offStage[22][0]);  algoMemHandler.S15_G  = asAlgo(offStage[22][1]);  algoMemHandler.S15_B  = asAlgo(offStage[22][2]);
    algoMemHandler.S16_R  = asAlgo(offStage[23][0]);  algoMemHandler.S16_G  = asAlgo(offStage[23][1]);  algoMemHandler.S16_B  = asAlgo(offStage[23][2]);
    algoMemHandler.S17_R  = asAlgo(offStage[24][0]);  algoMemHandler.S17_G  = asAlgo(offStage[24][1]);  algoMemHandler.S17_B  = asAlgo(offStage[24][2]);

    algoMemHandler.Scr_BlurA    = asAlgo(offBlurA);
    algoMemHandler.Scr_BlurB    = asAlgo(offBlurB);
    algoMemHandler.Scr_Luma     = asAlgo(offLuma);
    algoMemHandler.Scr_Field    = asAlgo(offField);
    algoMemHandler.Scr_FieldLo  = asAlgo(offFieldLo);
    algoMemHandler.Scr_Dbar     = asAlgo(offDbar);
    algoMemHandler.Scr_DbarBlur = asAlgo(offDbarBlur);

    algoMemHandler.Scr_LogE_R = asAlgo(offLogE[0]);
    algoMemHandler.Scr_LogE_G = asAlgo(offLogE[1]);
    algoMemHandler.Scr_LogE_B = asAlgo(offLogE[2]);

    algoMemHandler.Scr_Grain_R = asAlgo(offGrain[0]);
    algoMemHandler.Scr_Grain_G = asAlgo(offGrain[1]);
    algoMemHandler.Scr_Grain_B = asAlgo(offGrain[2]);


    return algoMemHandler;
}


// ---------------------------------------------------------------------------
//  free_memory_buffers
// ---------------------------------------------------------------------------
void free_memory_buffers (MemHandler& algoMemHandler) noexcept
{
    // Safe on an already-zeroed handler and safe to call twice: the null test
    // covers both, so a double free cannot reach the pool.
    if (nullptr != algoMemHandler.SuperBufferHead)
        FreeMemoryBlock(static_cast<int32_t>(algoMemHandler.memBlockId));

    // Zero the whole structure, so a stale pointer cannot be dereferenced after the
    // free. Assigning a fresh zero-initialised aggregate rather than clearing field
    // by field, so a field added to the struct later cannot be missed here.
    algoMemHandler = MemHandler{};

    return;
}
