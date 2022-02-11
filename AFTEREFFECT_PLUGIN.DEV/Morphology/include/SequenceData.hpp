#pragma once

#include <atomic>
#include <mutex>
#include <vector>
#include "SE_Interface.hpp"

namespace DataStore
{
	std::uint64_t addObjPtr2Container(SE_Interface* pObj);
	void disposeObjPtr(const std::uint64_t& idx = 0u);
	SE_Interface* getObject (const std::uint64_t& idx = 0u);
}