#pragma once

// ---------------------------------------------------------------------------
//  AlgoMemHandler.hpp
//
//  Arena memory for the film simulation engine.
//
//  ONE allocation per frame geometry. The block is obtained from the project's
//  memory manager, sliced into named pointers by a running offset stack, and
//  handed to the engine as a plain aggregate. The engine reads the pointers and
//  never allocates, resizes or frees anything.
//
//  BUFFER POLICY: TWO ALTERNATING STAGE TRIPLES, WITH FULL RETENTION AVAILABLE
//
//  Originally every stage wrote to its own dedicated buffer, retained for the whole
//  frame, so any intermediate could be dumped without re-running the chain. That was
//  the only practical way to validate a twenty-stage physical model against a
//  reference, and it did its job - several real bugs were localised that way.
//
//  It also cost 25 retained triples, which is 88 of the 94 planes in the arena and
//  the reason a 2816 x 1536 frame in double needed 3.15 GB and was refused.
//
//  The scalar path is now verified, so the chain alternates between TWO triples.
//  This is sound because the pipeline is a strict linear chain - every stage reads
//  only its immediate predecessor, verified across all 25 stage calls. The one
//  cross-stage lifetime in the engine is Scr_LogE_*, written at stage 7/8 and read
//  at 8b, and it has its own retained triple.
//
//  The saving is large: 88 AlgoType planes become 19, about 76 per cent in double
//  and 73 per cent in float. 4K DCI in double falls from 6.00 GB to 1.45 GB, which
//  fits the existing int32 pool interface with room to spare.
//
//  ALGO_RETAIN_ALL_STAGES restores the original layout for debugging. Both modes
//  must produce BIT-IDENTICAL output; that equality is the regression test for this
//  optimisation, and it means a future discrepancy can still be bisected against a
//  fully retained run.
//
//  Note what this is NOT: it is not in-place operation. Source and destination are
//  always different planes. Eleven stages read a neighbourhood of their source and
//  would break outright if the two were the same buffer.
//
//  WHY ONE PADDED GEOMETRY FOR BOTH ELEMENT TYPES
//
//  ImgType and AlgoType have different alignment quanta - float needs 8 elements
//  to span 32 bytes, double needs 4. Rather than carry two strides, the width is
//  padded to a multiple of the LARGER of the two, so a single padW serves both:
//  padW * sizeof(ImgType) and padW * sizeof(AlgoType) are then both multiples of
//  32, which puts every row start of every plane on a 32-byte boundary.
// ---------------------------------------------------------------------------

// Project-wide primitives, included unconditionally as required by the project
// coding standard.
#include "Common.hpp"
#include "CompileTimeUtils.hpp"

// ImgType, AlgoType, HighPrecType and the alignment constants. The only place
// the engine's numeric representations are chosen.
#include "AlgoTypes.hpp"

// FilmProfile and FilmFormat, for the caller-owned lookup tables below.
#include "film_profiles.hpp"

#include <cstddef>   // std::size_t
#include <cstdint>   // int32_t, int64_t, uint8_t


// ---------------------------------------------------------------------------
//  ALGO_RETAIN_ALL_STAGES -- give every stage its own retained triple.
//
//      0   two alternating triples. Production. 19 AlgoType planes.
//      1   25 dedicated triples, every intermediate survives. 88 planes.
//
//  COMPILE-TIME on purpose, not a runtime flag. A runtime switch would double the
//  pointer bookkeeping in the hot path for a facility only ever wanted during
//  development, and it would make the two layouts a property of a frame rather than
//  of a build - which is exactly the sort of thing that makes a bug irreproducible.
//
//  Set to 1 when a stage's intermediate needs inspecting, or to bisect a suspected
//  regression against a fully retained run. The two builds must agree bit for bit.
// ---------------------------------------------------------------------------
#ifndef ALGO_RETAIN_ALL_STAGES
  #define ALGO_RETAIN_ALL_STAGES 0
#endif

// ---------------------------------------------------------------------------
//  Physical stage triples actually allocated, as opposed to the 25 logical stages.
//
//  Derived from the switch rather than written twice, so the allocator's
//  slice-count cross-check cannot fall out of step with the layout it is checking.
// ---------------------------------------------------------------------------
#if ALGO_RETAIN_ALL_STAGES
  constexpr int32_t ALGO_PHYSICAL_STAGE_TRIPLES = 25;
#else
  constexpr int32_t ALGO_PHYSICAL_STAGE_TRIPLES = 2;
#endif


// ---------------------------------------------------------------------------
//  PADDING QUANTUM, in elements of the widest requirement
//
//  The larger of the two type quanta, so one padded width satisfies both. With
//  ImgType = float (8) and AlgoType = double (4) this is 8, giving row byte
//  counts that are multiples of 32 for both element sizes.
//
//  Expressed as a compile-time maximum rather than a literal 8, so that changing
//  either alias in AlgoTypes.hpp cannot silently break row alignment.
// ---------------------------------------------------------------------------
constexpr std::size_t ALGO_PAD_ELEMS =
    (IMG_ALIGN_ELEMS > ALGO_ALIGN_ELEMS) ? IMG_ALIGN_ELEMS : ALGO_ALIGN_ELEMS;


// ---------------------------------------------------------------------------
//  MemHandler
//
//  A plain aggregate. No constructors, no member functions, no inheritance: it
//  must stay trivially copyable so a per-thread copy is free, and it must hold no
//  state a concurrent invocation could mutate.
//
//  Zero-initialise with  MemHandler mh{};  before use.
//
//  THE FIRST TWO FIELDS ARE RESERVED FOR THE MEMORY MANAGER. The algorithm must
//  not read, write or reason about them. They exist so the block can be returned
//  to the pool by free_memory_buffers().
//
//  Field order below follows the pipeline, so the struct reads as a map of the
//  frame's progress and the arena layout mirrors it.
// ---------------------------------------------------------------------------
struct MemHandler
{
    // === RESERVED - memory manager only. Do not touch from the algorithm. ===
    int64_t  memBlockId;                    // pool handle for FreeMemoryBlock
    uint8_t* RESTRICT SuperBufferHead;      // base of the single allocation
    // ========================================================================

    int32_t padW;        // padded width, elements. Row stride for every plane.
    int32_t padH;        // padded height, rows
    int32_t activeW;     // requested width; the meaningful part of each row
    int32_t activeH;     // requested height

    // --- caller-owned lookup tables, NOT arena memory ----------------------
    //
    // The engine resolves a film profile and a gauge from AlgoControls, so it needs
    // both tables. It cannot call film::GetFilmDatabase() itself: that returns a
    // std::vector BY VALUE and would allocate on every frame, which the
    // no-allocation rule forbids.
    //
    // alloc_memory_buffers() FILLS THESE FOUR FIELDS ITSELF from tables it builds
    // once, on first use, and never modifies. The caller does not have to do
    // anything, and cannot forget to.
    //
    // An earlier revision left it to the caller to attach them afterwards. That was
    // a mistake: they defaulted to null, the engine is forbidden from validating
    // them, and missing the step produced a null dereference on the first frame
    // with no diagnostic at all. Anything the engine must have and cannot check
    // belongs to the allocator.
    //
    // Const because the engine only ever reads them.
    const film::FilmProfile* RESTRICT pProfileDb;
    int32_t                           profileCount;
    const film::FilmFormat*  RESTRICT pFormatDb;
    int32_t                           formatCount;

    // Print stocks. Needed by the anchor solve at stage 8 for a NEGATIVE stock,
    // whose free parameter is the print exposure offset and so cannot be found
    // without knowing what it will be printed onto. A reversal stock has no print
    // stage and never reads this table.
    //
    // Attached by the allocator for exactly the same reason as the two above: the
    // engine cannot allocate it itself and is forbidden from checking it.
    const film::PrintStock*  RESTRICT pPrintDb;
    int32_t                           printCount;

    // --- boundary images, STORAGE type -------------------------------------
    // Scene-linear source handed in by the caller, read only by stage 2, and the
    // display-linear destination written only by the final stage. These are the
    // two points where ImgType and AlgoType meet.
    ImgType* RESTRICT Src_R;
    ImgType* RESTRICT Src_G;
    ImgType* RESTRICT Src_B;
    ImgType* RESTRICT Dst_R;
    ImgType* RESTRICT Dst_G;
    ImgType* RESTRICT Dst_B;

    // --- stage outputs, exposure domain ------------------------------------
    AlgoType* RESTRICT S02_R;   AlgoType* RESTRICT S02_G;   AlgoType* RESTRICT S02_B;   // relative exposure
    AlgoType* RESTRICT S02b_R;  AlgoType* RESTRICT S02b_G;  AlgoType* RESTRICT S02b_B;  // taking filters
    AlgoType* RESTRICT S03_R;   AlgoType* RESTRICT S03_G;   AlgoType* RESTRICT S03_B;   // colour balance
    AlgoType* RESTRICT S03b_R;  AlgoType* RESTRICT S03b_G;  AlgoType* RESTRICT S03b_B;  // veiling flare
    AlgoType* RESTRICT S03c_R;  AlgoType* RESTRICT S03c_G;  AlgoType* RESTRICT S03c_B;  // temporal flicker [not yet written]
    AlgoType* RESTRICT S04_R;   AlgoType* RESTRICT S04_G;   AlgoType* RESTRICT S04_B;   // vignette x coating field
    AlgoType* RESTRICT S05_R;   AlgoType* RESTRICT S05_G;   AlgoType* RESTRICT S05_B;   // halation
    AlgoType* RESTRICT S06_R;   AlgoType* RESTRICT S06_G;   AlgoType* RESTRICT S06_B;   // emulsion MTF
    AlgoType* RESTRICT S06b_R;  AlgoType* RESTRICT S06b_G;  AlgoType* RESTRICT S06b_B;  // corner defocus
    AlgoType* RESTRICT S07_R;   AlgoType* RESTRICT S07_G;   AlgoType* RESTRICT S07_B;   // monochrome collapse

    // --- stage outputs, density domain -------------------------------------
    AlgoType* RESTRICT S08_R;   AlgoType* RESTRICT S08_G;   AlgoType* RESTRICT S08_B;   // characteristic curve
    AlgoType* RESTRICT S08b_R;  AlgoType* RESTRICT S08b_G;  AlgoType* RESTRICT S08b_B;  // interimage effects
    AlgoType* RESTRICT S09_R;   AlgoType* RESTRICT S09_G;   AlgoType* RESTRICT S09_B;   // DIR coupler lateral
    AlgoType* RESTRICT S09b_R;  AlgoType* RESTRICT S09b_G;  AlgoType* RESTRICT S09b_B;  // negative-side defects [not yet written]
    AlgoType* RESTRICT S10_R;   AlgoType* RESTRICT S10_G;   AlgoType* RESTRICT S10_B;   // scan MTF + misregistration
    AlgoType* RESTRICT S10b_R;  AlgoType* RESTRICT S10b_G;  AlgoType* RESTRICT S10b_B;  // narrow-gauge edge fog
    AlgoType* RESTRICT S11_R;   AlgoType* RESTRICT S11_G;   AlgoType* RESTRICT S11_B;   // grain
    AlgoType* RESTRICT S12_R;   AlgoType* RESTRICT S12_G;   AlgoType* RESTRICT S12_B;   // dye crosstalk
    AlgoType* RESTRICT S13_R;   AlgoType* RESTRICT S13_G;   AlgoType* RESTRICT S13_B;   // dupe generations, then print

    // --- stage outputs, display domain -------------------------------------
    AlgoType* RESTRICT S14_R;   AlgoType* RESTRICT S14_G;   AlgoType* RESTRICT S14_B;   // transmittance
    AlgoType* RESTRICT S14b_R;  AlgoType* RESTRICT S14b_G;  AlgoType* RESTRICT S14b_B;  // reseau reconstruction
    AlgoType* RESTRICT S14c_R;  AlgoType* RESTRICT S14c_G;  AlgoType* RESTRICT S14c_B;  // silver image tone
    AlgoType* RESTRICT S15_R;   AlgoType* RESTRICT S15_G;   AlgoType* RESTRICT S15_B;   // gate weave [not yet written]
    AlgoType* RESTRICT S16_R;   AlgoType* RESTRICT S16_G;   AlgoType* RESTRICT S16_B;   // gate-side defects [not yet written]
    AlgoType* RESTRICT S17_R;   AlgoType* RESTRICT S17_G;   AlgoType* RESTRICT S17_B;   // final clamp

    // --- transient scratch -------------------------------------------------
    // Nothing here is worth keeping past the stage that used it, but it is still
    // arena-resident because the engine may not allocate. Listed apart from the
    // retained chain so a later memory pass can pool it without touching stage
    // outputs.
    AlgoType* RESTRICT Scr_BlurA;      // separable blur intermediate
    AlgoType* RESTRICT Scr_BlurB;      // second half of a two-pass blur
    AlgoType* RESTRICT Scr_Luma;       // luminance driving the veiling flare
    AlgoType* RESTRICT Scr_Field;      // vignette x coating multiplier field
    AlgoType* RESTRICT Scr_FieldLo;    // low-resolution coating field, corner use
    AlgoType* RESTRICT Scr_Dbar;       // mean of the three density channels
    AlgoType* RESTRICT Scr_DbarBlur;   // blurred copy of that mean

    // Log exposure, retained across the characteristic curve into the interimage
    // fixed point: stage 8b needs the ORIGINAL log exposure, not the densities.
    AlgoType* RESTRICT Scr_LogE_R;
    AlgoType* RESTRICT Scr_LogE_G;
    AlgoType* RESTRICT Scr_LogE_B;

    // Per-channel grain field before density scaling.
    AlgoType* RESTRICT Scr_Grain_R;
    AlgoType* RESTRICT Scr_Grain_G;
    AlgoType* RESTRICT Scr_Grain_B;

    std::size_t totalSize;   // bytes obtained from the pool
};


// ---------------------------------------------------------------------------
//  NO VIEW OR WRAPPER TYPES
//
//  Deliberately absent. The engine works on RAW POINTERS, and every piece of
//  geometry - active width, active height, row pitch - is passed to each stage as
//  an explicit parameter rather than carried inside an object.
//
//  Two reasons, both practical rather than stylistic:
//
//    - A stage signature that takes a pointer, a width, a height and a pitch
//      states exactly what it touches. There is no indirection to chase and no
//      question about which geometry a wrapper is carrying.
//    - The eventual AVX2 path wants the pointer and the stride as plain values in
//      registers. A wrapper is free at -O2 but it obscures the loop bounds the
//      vectoriser has to reason about, and it makes RESTRICT harder to apply where
//      it matters.
//
//  Row addressing is therefore the explicit form used throughout:
//
//      ptr + static_cast<std::ptrdiff_t>(y) * pitch
//
//  where pitch is padW, in ELEMENTS. Because the padded width satisfies the larger
//  of the two alignment quanta, ONE pitch value is correct for every plane of
//  either element type - there is no separate ImgType and AlgoType stride to keep
//  straight.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
//  alloc_memory_buffers
//
//  Compute the offset stack for the given frame geometry, obtain one block from
//  the pool, map every pointer into it, and return the filled structure.
//
//  On failure - a degenerate geometry, or the pool refusing the request - the
//  returned structure is zeroed, so SuperBufferHead is null and totalSize is
//  zero. Test that before use; there is no exception and no error code.
//
//  dbgPrn: print the layout and per-buffer offsets. Diagnostic only, off in
//  production, and it writes nothing when false.
// ---------------------------------------------------------------------------
MemHandler alloc_memory_buffers(const int32_t sizeX, const int32_t sizeY);


// ---------------------------------------------------------------------------
//  free_memory_buffers
//
//  Return the block to the pool and zero the structure, so a stale pointer
//  cannot be dereferenced after the free. Safe on an already-zeroed handler and
//  safe to call twice.
// ---------------------------------------------------------------------------
void free_memory_buffers (MemHandler& algoMemHandler) noexcept;

inline bool mem_handler_valid(const MemHandler& hndl) noexcept
{
    return (hndl.memBlockId >= 0 && hndl.SuperBufferHead != nullptr) ? true : false;
}
