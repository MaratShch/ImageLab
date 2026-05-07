#include "AlgorithmMain.hpp"

void Algorithm_Main
(
    const MemHandler& memHandler,
    const int32_t sizeX, 
    const int32_t sizeY, 
    const AlgoControls& algoCtrl
)
{
    // ========================================================================
    // 1. Sanitize UI Inputs
    // ========================================================================
    AlgoControls ctrl = algoCtrl;
    ctrl.Sanitize(); // Ensures iterations is safely between 1 and 4

    // ========================================================================
    // 2. Ping-Pong Pointer Setup
    // ========================================================================
    // Source always starts as the original unpacked image.
    float* currentSource = memHandler.proc_Y;

    // We pre-calculate the first target based on Odd/Even iterations.
    // If iterations is ODD (1, 3), we want to start by writing to out_Y.
    // If iterations is EVEN (2, 4), we want to start by writing to scratch_Y.
    float* currentTarget = (ctrl.iterations % 2 != 0) ? memHandler.out_Y : memHandler.scratch_Y;

    // ========================================================================
    // 3. The Iteration Loop
    // ========================================================================
    for (int32_t i = 0; i < ctrl.iterations; ++i)
    {
        // STEP A: Pad the current source buffer.
        // Even if the source is out_Y or scratch_Y from a previous pass, 
        // they have the invisible halo allocated, so this is perfectly safe.
        ApplyMirroredPadding
        (
            currentSource, 
            sizeX, 
            sizeY, 
            memHandler.strideY_Elements, 
            ctrl.radius
        );

        // STEP B: Safety Catch
        // Force the absolute final iteration to write to out_Y, just in case.
        if (i == ctrl.iterations - 1) 
        {
            currentTarget = memHandler.out_Y;
        }

        // STEP C: Process the image
        // We temporarily cast currentSource to const to guarantee the math 
        // engine doesn't alter the buffer it's reading from.
        ProcessImage_Scalar
        (
            const_cast<const float*>(currentSource), 
            currentTarget, 
            sizeX, 
            sizeY, 
            memHandler.strideY_Elements, 
            ctrl
        );

        // STEP D: Swap Pointers for the Next Pass
        // The output of this pass becomes the input of the next pass!
        currentSource = currentTarget;

        // The new target is whichever intermediate buffer we AREN'T currently using.
        if (currentSource == memHandler.scratch_Y) 
        {
            currentTarget = memHandler.out_Y;
        } 
        else 
        {
            currentTarget = memHandler.scratch_Y;
        }
    }
    
    return;
}