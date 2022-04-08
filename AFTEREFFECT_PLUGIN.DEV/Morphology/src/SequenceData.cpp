#include "SequenceData.hpp"

namespace DataStore
{
	static std::mutex gGlobalProtect;
	static std::vector<SE_Interface*> gSeInterfase;

	std::uint64_t addObjPtr2Container(SE_Interface* pObj)
	{
		std::lock_guard<std::mutex> lock(gGlobalProtect);
		gSeInterfase.push_back(pObj);
		return (gSeInterfase.size() - 1u);
	}


	void disposeObjPtr(const std::uint64_t& idx)
	{
		std::lock_guard<std::mutex> lock(gGlobalProtect);
		if (gSeInterfase.size() > idx)
			gSeInterfase.erase(gSeInterfase.begin() + idx);
		return;
	}


	SE_Interface* getObject(const std::uint64_t& idx)
	{
		std::lock_guard<std::mutex> lock(gGlobalProtect);
		return (gSeInterfase.size() > idx) ? gSeInterfase[idx] : nullptr;
	}
};