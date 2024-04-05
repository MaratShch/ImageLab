#include "AlgoColorDescriptor.hpp"

std::atomic<CAlgoColorDescriptor<ColorDescriptorT>*> CAlgoColorDescriptor<ColorDescriptorT>::s_instance;
std::mutex CAlgoColorDescriptor<ColorDescriptorT>::s_protectMutex;