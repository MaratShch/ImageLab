#ifndef _IMAGE_LAB_COLOR_TEMPERATURE_DRAW_APIS__
#define _IMAGE_LAB_COLOR_TEMPERATURE_DRAW_APIS__

#include "AlgoRules.hpp"
#include "cct_interface.hpp"

void StartGuiThread(void);
void StopGuiThread (void);
void ForceRedraw   (void) noexcept;
bool isRedraw      (void) noexcept;
bool ProcRedrawComplete(void) noexcept;

void SetGUI_CCT (const std::pair<AlgoProcT, AlgoProcT>& cct_duv) noexcept;


#endif // _IMAGE_LAB_COLOR_TEMPERATURE_DRAW_APIS__